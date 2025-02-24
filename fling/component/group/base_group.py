import copy
import time
import torch

from fling.utils import get_params_number
from fling.utils.compress_utils import fed_avg
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger, get_weights
from fling.component.client import ClientTemplate
from functools import reduce

@GROUP_REGISTRY.register('base_group')
class ParameterServerGroup:
    r"""
    Overview:
        Base implementation of the group in federated learning.
    """

    def __init__(self, args: dict, logger: Logger):
        r"""
        Overview:
            Lazy initialization of group.
            To complete the initialization process, please call `self.initialization()` after server and all clients
        are initialized.
        Arguments:
            - args: arguments in dict type.
            - logger: logger for this group
        Returns:
            - None
        """
        self.clients = []
        self.server = None
        self.args = args
        self.logger = logger
        self._time = time.time()

    def initialize(self) -> None:
        r"""
        Overview:
            In this function, several things will be done:
            1) Set ``fed_key`` in each client is determined, determine which parameters should be included for federated
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
        fed_keys = get_weights(
            self.clients[0].model, self.args.group.aggregation_parameters, return_dict=True, include_non_param=True
        ).keys()

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}

        # Resume from the checkpoint if needed.
        if self.args.other.resume_path is not None:
            sd = dict(torch.load(self.args.other.resume_path))
            for k, v in sd.items():
                if k in glob_dict.keys():
                    glob_dict[k] = v
        self.server.glob_dict = glob_dict

        self.set_fed_keys()

        # Step 3.
        self.sync()

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) * 4 / 1e6)
        )

    def append(self, client: ClientTemplate) -> None:
        r"""
        Overview:
            Append a client into the group.
        Arguments:
            - client: client to be added.
        Returns:
            - None
        """
        self.clients.append(client)

    def aggregate(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        r"""
        Overview:
            Aggregate all client models.
        Arguments:
            - train_round: current global epochs.
            - aggr_parameter_args: What parameters should be aggregated. If set to ``None``, the initialized setting \
                will be used.
        Returns:
            - trans_cost: uplink communication cost.
        """
        # Pick out the parameters for aggregation if needed.
        if aggr_parameter_args is not None:
            fed_keys_bak = self.clients[0].fed_keys
            new_fed_keys = get_weights(
                self.clients[0].model, aggr_parameter_args, return_dict=True, include_non_param=True
            ).keys()
            for client in self.clients:
                client.set_fed_keys(new_fed_keys)

        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        else:
            raise KeyError('Unrecognized compression method: ' + self.args.group.aggregation_method)

        # Add logger for time per round.
        # This time is the interval between two times of executing this ``aggregate()`` function.
        time_per_round = time.time() - self._time
        self._time = time.time()
        self.logger.add_scalar('time/time_per_round', time_per_round, train_round)

        if aggr_parameter_args is not None:
            for client in self.clients:
                client.set_fed_keys(fed_keys_bak)

        return trans_cost



    def flush(self) -> None:
        r"""
        Overview:
            Reset this group and clear all server and clients.
        Returns:
            - None
        """
        self.clients = []
        self.server = None

    def sync(self) -> None:
        r"""
        Overview:
            Synchronize all local models, making their parameters same as global model.
        Returns:
            - None
        """
        state_dict = self.server.glob_dict
        for client in self.clients:
            client.update_model(state_dict)

    def set_fed_keys(self) -> None:
        r"""
        Overview:
            Set `fed_keys` of each client, determine which parameters should be included for federated learning
        Returns:
            - None
        """
        for client in self.clients:
            client.set_fed_keys(self.server.glob_dict.keys())

    # TODO：收到每个客户端的domain-style信息，随机打乱顺序，并传入到各自client的模型当中
    def transfer_domainspecific_information(self, domain_information, participated_clients):
        n = len(domain_information)
        transfer_clients = torch.randperm(n)
        for client_idx in range(len(participated_clients)):
            transfer_client_idx = transfer_clients[client_idx]
            origin_client_idx = participated_clients[client_idx]
            self.clients[origin_client_idx].model.crossclient_style(domain_information[transfer_client_idx])

    # TODO: Aggregate prompt
    def aggregate_prompt(self) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        self.server.glob_prompter_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.prompter.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].prompter.state_dict()
        }
        state_dict = self.server.glob_prompter_dict
        for client in self.clients:
            client.update_prompt(state_dict)
        # Calculate the ``trans_cost``.
        trans_cost = 0
        state_dict = self.clients[0].prompter.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost

    def fedsdr_aggregate_global_variant(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_variant_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.global_variant_model.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].global_variant_model.state_dict()
        }
        for client in self.clients:
            client.global_variant_model.load_state_dict(global_variant_dict)
        trans_cost = 0
        state_dict = self.clients[0].global_variant_model.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost

    def fedsdr_aggregate_environmentList(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        group_num = len(self.clients[0].environmentList)
        environmentList = [copy.deepcopy(envir_classifier) for envir_classifier in self.clients[0].environmentList]
        for group in range(group_num):
            sample_all = 0.
            paramlist = self.clients[0].environmentList[group].state_dict()
            for name in paramlist.keys():
                for client_idx in range(len(self.clients)):
                    if client_idx == 0:
                        paramlist[name] = self.clients[client_idx].environmentList[group].state_dict()[name] * self.clients[client_idx].group_num[group]
                    else:
                        paramlist[name] += self.clients[client_idx].environmentList[group].state_dict()[name] * \
                                          self.clients[client_idx].group_num[group]
                    sample_all += self.clients[client_idx].group_num[group]
                paramlist[name] /= sample_all
            environmentList[group].load_state_dict(paramlist)

        for client in self.clients:
            for group in range(group_num):
                client.environmentList[group].load_state_dict(environmentList[group].state_dict())
        trans_cost = 0
        state_dict = self.clients[0].environmentList[0].state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        return trans_cost * 4 * group_num
    ###############################################################################################################################

    def aggregate_auxcls(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_auxcls_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.auxilary_classifier.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].auxilary_classifier.state_dict()
        }
        for client in self.clients:
            client.auxilary_classifier.load_state_dict(global_auxcls_dict)
        trans_cost = 0
        state_dict = self.clients[0].auxilary_classifier.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost

    def aggregate_global_invariant(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_variant_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.global_invariant_model.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].global_invariant_model.state_dict()
        }

        return global_variant_dict

    def aggregate_global_fedpin(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_variant_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.global_invariant_model.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].global_invariant_model.state_dict()
        }
        return global_variant_dict

    def aggregate_aux_classifier(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        aux_classifier_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.auxilary_classifier.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].auxilary_classifier.state_dict()
        }
        return aux_classifier_dict



    def aggregate_generator(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        generator_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.generator.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].generator.state_dict()
        }
        for client in self.clients:
            client.generator.load_state_dict(generator_dict)
        trans_cost = 0
        state_dict = self.clients[0].generator.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost

    ############ FedMutual ##############################
    def aggregate_probe(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_g1net_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.g1_net.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].g1_net.state_dict()
        }
        for client in self.clients:
            client.g1_net.load_state_dict(global_g1net_dict)
        trans_cost = 0
        state_dict = self.clients[0].g1_net.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost

    def aggregate_aux(self, train_round: int, aggr_parameter_args: dict = None) -> int:
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        global_g2net_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.sample_num / total_samples * client.g2_net.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].g2_net.state_dict()
        }
        for client in self.clients:
            client.g2_net.load_state_dict(global_g2net_dict)
        trans_cost = 0
        state_dict = self.clients[0].g2_net.state_dict()
        for k in state_dict:
            trans_cost += len(self.clients) * state_dict[k].numel()
        # 1B = 32bit
        return 4 * trans_cost