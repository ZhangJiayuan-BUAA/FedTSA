from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY

import numpy as np

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


@DATASET_REGISTRY.register('cifar100_test')
class CIFAR100_TestDataset(Dataset):

    def __init__(self, cfg, train=False):
        super(CIFAR100_TestDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.common_corruptions = cfg.data.corruption
        transform = get_data_transform(cfg.data.transforms, train=train)
        self._prepare_test_data(train, transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}

    def _prepare_test_data(self, train, transform):
        tesize = 10000
        if self.cfg.data.corruption is None or train:
            print('Test on the original test set')
            self.dataset = CIFAR100(self.cfg.data.data_path, train=train, transform=transform, download=True)
        elif self.cfg.data.corruption in common_corruptions:
            print('Test on %s level %d' % (self.cfg.data.corruption, self.cfg.data.level))
            teset_raw = np.load(self.cfg.data.data_path + '/CIFAR-100-C/%s.npy' % (self.cfg.data.corruption))
            teset_raw = teset_raw[(self.cfg.data.level - 1) * tesize: self.cfg.data.level * tesize]
            self.dataset = CIFAR100(self.cfg.data.data_path, train=train, transform=transform, download=True)
            self.dataset.data = teset_raw
        else:
            raise "Don't have this type of data!"

    def _prepare_raw_test_data(self, train):
        tesize = 10000
        if self.cfg.data.corruption is None or train:
            print('Test on the original test set')
            self.dataset_strong = CIFAR100(self.cfg.data.data_path, train=train, download=True)
        elif self.cfg.data.corruption in common_corruptions:
            print('Test on %s level %d' % (self.cfg.data.corruption, self.cfg.data.level))
            teset_raw = np.load(self.cfg.data.data_path + '/CIFAR-100-C/%s.npy' % (self.cfg.data.corruption))
            teset_raw = teset_raw[(self.cfg.data.level - 1) * tesize: self.cfg.data.level * tesize]
            self.dataset_strong = CIFAR100(self.cfg.data.data_path, train=train, download=True)
            self.dataset_strong.data = teset_raw
        else:
            raise "Don't have this type of data!"
