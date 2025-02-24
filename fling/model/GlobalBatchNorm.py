import torch
import torch.nn as nn

class GlobalBatchNorm(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, ratio=0.5):
        super(GlobalBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.sum_num = 0.
        self.sample_num = 0.
        self.ratio = ratio
        self.bank_mean = self.running_mean.view(1, -1)
        self.bank_var = self.running_var.view(1, -1)
        self.weighted_mean, self.weighted_var = [], []

    def update(self, mean, var):
        self.mean = mean
        self.var = var

    def update_sample_num(self, sample_num):
        self.sample_num = sample_num

    def forward(self, input):                             # input(N,C,H,W)
        if self.training:
            self.batch_mean = input.mean([0, 2, 3])
            self.batch_var = input.var([0, 2, 3], unbiased=False)
            self.mean = self.batch_mean
            self.var = self.batch_var
        else:
            if self.weighted_mean == []:
                self.batch_mean = input.mean([0, 2, 3])
                self.batch_var = input.var([0, 2, 3], unbiased=False)
                self.mean = self.batch_mean
                self.var = self.batch_var
            else:
                self.mean = self.weighted_mean
                self.var = self.weighted_var

        input = self.alpha * (input - self.mean[None, :, None, None]) / (torch.sqrt(self.var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


    def update_prior(self, sample_num):
        if self.past_mean == 0:
            self.past_mean = self.batch_mean
            self.past_var = self.batch_var
        else:
            self.past_mean = (self.past_mean * self.sum + self.mean * sample_num) / (self.sum + sample_num)
            self.past_var = (self.past_var * self.sum + self.var * sample_num) / (self.sum + sample_num)
        self.sum += sample_num

# Define a custom ResNet-8 model
class CustomResNeXt(nn.Module):
    def __init__(self, pretrained_backbone, alpha=0.5):
        super(CustomResNeXt, self).__init__()
        # Load the pretrained backbone and initialize alpha
        self.backbone = pretrained_backbone
        self.alpha = nn.Parameter(torch.tensor([alpha for _ in range(31)]))

        # Replace BatchNorm layers with CustomBatchNorm2d
        replace_mods = self.replace_batchnorm(self.backbone)
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)

    def replace_batchnorm(self, parent):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = GlobalBatchNorm(child.num_features, child.eps, child.momentum, affine=child.affine,
                                  track_running_stats=child.track_running_stats)
                module.running_mean = child.running_mean
                module.running_var = child.running_var
                module.weight = child.weight
                module.bias = child.bias
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(self.replace_batchnorm(child))

        return replace_mods

    def forward(self, x, mode='compute-logit'):
        return self.backbone(x, mode)