import torch
import torch.nn.functional as F

from fling.utils import get_params_number
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import VariableMonitor
from fling.model.stattention import ST_block, SpatialAttention
from fling.component.group import ParameterServerGroup

@GROUP_REGISTRY.register('adapt_group')
class TTAServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Base implementation of the group in federated learning.
    """

    def __init__(self, args: dict, logger: VariableMonitor):
        r"""
        Overview:
            Implementation of the group in FedTTA
        """
        super(TTAServerGroup, self).__init__(args, logger)
        self.history_feature = []
        self.history_weight = [[] for _ in range(args.client.client_num)]
        self.indicator = torch.tensor([])
        self.time_slide = 10
        self.collaboration_graph = []


    def initialize(self) -> None:
        r"""
        Overview:
            In this function, several things will be done:
            1) Set ``fed_key`` in each client is determined, determine which parameters shoul be included for federated
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
        if self.args.group.aggregation_parameters.name == 'all':
            fed_keys = self.clients[0].model.state_dict().keys()
        elif self.args.group.aggregation_parameters.name == 'contain':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'except':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(self.clients[0].model.state_dict().keys()) - set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'include':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for name, param in self.clients[0].model.named_parameters():
                if keywords in name:
                    fed_keys.append(name)
            fed_keys = list(set(fed_keys))
        else:
            raise ValueError(f'Unrecognized aggregation_parameters.name: {self.args.group.aggregation_parameters.name}')

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}
        self.server.glob_dict = glob_dict
        self.set_fed_keys()

        # Step 3.
        if not self.args.other.resume:
            self.sync()

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) / 1e6)
        )

    def st_agg_bn(self, time_att=None, space_att=None, global_mean=None, wotime=False):
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        for chosen_layer in range(n_chosen_layer):
            if self.history_feature[0].shape[1] < self.time_slide:
                feature_input = self.history_feature[chosen_layer][:, :, :]
            else:
                T_all = self.history_feature[chosen_layer].shape[1]
                feature_input = self.history_feature[chosen_layer][:, T_all - self.time_slide:, :]
            N, T, D = feature_input.shape
            out = torch.matmul(space_att, feature_input.view(N, T, 1, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, D)

            half = out.shape[2] // 2
            for cidx in range(client_num):
                sum_mean[cidx][chosen_layer] = out[cidx][T - 1][:half]
                sum_var[cidx][chosen_layer] = out[cidx][T - 1][half:]
        return sum_mean, sum_var

    def aggregate_bn(self, train_round, global_mean, feature_indicator):
        # Store feature mean and variance
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        if len(self.history_feature) == 0:
            self.history_feature = [[] for _ in range(n_chosen_layer)]
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = feature_t.unsqueeze(1)
        else:
            for chosen_layer in range(n_chosen_layer):
                feature_t = []
                for cidx in range(client_num):
                    feature_t.append(global_mean[cidx][chosen_layer])
                feature_t = torch.stack(feature_t, dim=0)
                self.history_feature[chosen_layer] = torch.cat([self.history_feature[chosen_layer], feature_t.unsqueeze(1)], dim=1)

        # calculate aggregation rate & aggregate model weight
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        if self.args.group.aggregation_method == 'st':
            time_att, space_att = self.ST_attention(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)
        else:
            total_samples = float(sum([client.sample_num for client in self.clients]))
            for chosen_layer in range(len(global_mean[0])):
                half = len(global_mean[0][chosen_layer]) // 2
                for idx in range(len(global_mean)):
                    if idx == 0:
                        sum_mean[0][chosen_layer] = global_mean[idx][chosen_layer][:half] * self.clients[idx].sample_num
                        sum_var[0][chosen_layer] = global_mean[idx][chosen_layer][half:] * self.clients[idx].sample_num
                    else:
                        sum_mean[0][chosen_layer] += global_mean[idx][chosen_layer][:half] * self.clients[idx].sample_num
                        sum_var[0][chosen_layer] += global_mean[idx][chosen_layer][half:] * self.clients[idx].sample_num
            for idx in range(len(global_mean)):
                for chosen_layer in range(n_chosen_layer):
                    if idx == 0:
                        sum_mean[idx][chosen_layer] /= total_samples
                        sum_var[idx][chosen_layer] /= total_samples
                    else:
                        sum_mean[idx][chosen_layer] = sum_mean[0][chosen_layer]
                        sum_var[idx][chosen_layer] = sum_var[0][chosen_layer]

        for cidx in range(client_num):
            self.clients[cidx].update_bnstatistics(sum_mean[cidx], sum_var[cidx])

    def ST_attention(self, feature_indicator, wotime=False):
        '''
        :param feature_indicator: global_mean[ cidx ][ chosen_layer ][ D(mean) ]
        :return: weight1, weight2
        '''

        feature_indicator = torch.stack(feature_indicator, dim=0)
        if self.indicator.shape[0] == 0:
            self.indicator = feature_indicator.unsqueeze(1)
        else:
            self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        # Get Aggregate Weights with Trainable Modules
        self.time_slide = self.args.other.time_slide
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, :, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1]-self.time_slide:, :]
        ST_model = ST_block(args=self.args, dim=feature_input.shape[2])
        ST_model.cuda()
        opt = torch.optim.Adam(ST_model.parameters(), lr=self.args.other.st_lr)
        loss_min = 1000000
        epoch_num = self.args.other.st_epoch
        for epoch in range(epoch_num):
            # print('Epoch {}'.format(epoch))
            ST_model.train()
            logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input, wotime=wotime)
            loss_reg = F.mse_loss(feature_input, logits)
            loss_consist = F.mse_loss(logits, mask_logits)
            loss_robust = F.mse_loss(logits, aug_logits)

            loss = (loss_reg + self.args.other.robust_weight * loss_robust)

            if loss.item() < loss_min:
                time_att = t_sim
                space_att = s_sim
                loss_min = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.cuda.empty_cache()
        return time_att, space_att



