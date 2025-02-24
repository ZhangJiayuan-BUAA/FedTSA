import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from .client_template import ClientTemplate
from fling.model import get_model
from fling.utils.utils import VariableMonitor
from fling.model.GlobalBatchNorm import CustomResNeXt

@CLIENT_REGISTRY.register('fedtta_client')
class FedTTAClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedTTAClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.args = args
        self.client_id = client_id
        self.class_number = args.data.class_number
        self.adapt_iters = 1
        self.model = get_model(args)

    def init_weight(self, ckpt):
        # load state dict
        self.model.load_state_dict(ckpt)
        self.model_state = ckpt
        self.model_anchor = copy.deepcopy(self.model)
        # replace Custom Batch Normalization with Self-defined BN
        self.model = CustomResNeXt(self.model)

        self.chosen_layers = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.chosen_layers.append(m)
        self.n_chosen_layers = len(self.chosen_layers)

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}

    def test_source(self, test_data):
        self.adapt_loader = DataLoader(test_data, batch_size=self.args.learn.batch_size, shuffle=False)

        self.model_anchor.eval()
        self.model_anchor.to(self.device)
        self.model_anchor.requires_grad_(False)

        self.sample_num = len(test_data)
        monitor = VariableMonitor()
        criterion = nn.CrossEntropyLoss()
        # Main test loop.
        with torch.no_grad():
            for _, data in enumerate(self.adapt_loader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

                feature, out = self.model_anchor(batch_x, mode='compute-feature-logit')
                y_pred = torch.argmax(out, dim=-1)

                feature_mean = feature.mean(dim=0)
                feature_indicator = feature_mean

                loss = criterion(out, batch_y)
                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return mean_monitor_variables, feature_indicator

    def update_bnstatistics(self, clean_mean, clean_var):
        self.global_mean = clean_mean
        self.global_var = clean_var
        idx = 0
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weighted_mean = clean_mean[idx]
                m.weighted_var = clean_var[idx]
                idx += 1

    def adapt(self, test_data, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        self.sample_num = len(test_data)

        # Turn on model grads. collect_params
        self.model.requires_grad_(False)
        self.model.train()

        # Get Local TTA
        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()
        with torch.no_grad():
            for eps in range(self.adapt_iters):
                self.clean_mean = []
                for idx, data in enumerate(self.adapt_loader):
                    preprocessed_data = self.preprocess_data(data)
                    batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

                    out = self.model(batch_x)
                    y_pred = torch.argmax(out, dim=-1)

                    loss = criterion(out, batch_y)
                    monitor.append(
                        {
                            'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                            'test_loss': loss.item()
                        },
                        weight=preprocessed_data['y'].shape[0]
                    )

                for nm, m in self.model.named_modules():
                    if isinstance(m, nn.BatchNorm2d):
                        self.clean_mean.append(torch.cat([m.batch_mean, m.batch_var], dim=0))

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return self.clean_mean, mean_monitor_variables

    def inference(self, classifier=None, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()

        with torch.no_grad():
            for _, data in enumerate(self.adapt_loader):
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                outputs = self.model(batch_x)

                y_pred = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, batch_y)
                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return mean_monitor_variables



