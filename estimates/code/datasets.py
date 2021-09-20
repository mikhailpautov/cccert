import os
from typing import *
import math
import random

import torch
import torch.distributions as dist

import torchvision
from torchvision import transforms, datasets
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
CIFAR10_DIR = "/data/cifar10"



# list of all datasets 
DATASETS = ["mnist", "cifar10"]

LOC = torch.tensor(0.)
# CON = torch.tensor(4.)
CON = torch.tensor(1.6165)


class vonMisesRotate:
    '''
	Create Rotation transform with angles sampled from
        the von Mises(`loc`,`con`) distribution
    '''
    def __init__(self, loc, con):
        self.vm = dist.von_mises.VonMises(loc, con)

    def __call__(self, x):
        theta = self.vm.sample().item()
        return transforms.functional.rotate(x, theta*180/math.pi,
                                            InterpolationMode.BILINEAR)
class GaussianRotate:
    '''
	Create Rotation transform with angles sampled from
        the von Mises(`loc`,`con`) distribution
    '''
    def __init__(self, loc, con):
        self.gauss = dist.normal.Normal(loc, con)

    def __call__(self, x):
        theta = self.gauss.sample().item()
        return transforms.functional.rotate(x, theta*180/math.pi,
                                            InterpolationMode.BILINEAR)



# +
CIFAR_ROOT = '../../raid/datasets/cifar10/'
MNIST_ROOT = '../../raid/datasets/'

def get_dataloader(dataset:str, split:str, batch_size: int, CIFAR_ROOT=CIFAR_ROOT, MNIST_ROOT=MNIST_ROOT):
    print('specify path to the corresponding root folder in datasets.py, to be fixed later')
    if dataset == 'cifar10':
        data_root = CIFAR_ROOT
        if split == 'train':
            train_dataset = torchvision.datasets.CIFAR10(data_root, download=False, transform=torchvision.transforms.ToTensor())
            random_indices = random.sample(range(0, len(train_dataset)), 500)
            train_dataset = torch.utils.data.Subset(train_dataset, random_indices)
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        elif split == 'test':
            test_dataset = torchvision.datasets.CIFAR10(data_root, train=False, download=False, transform=torchvision.transforms.ToTensor())
            random_indices = random.sample(range(0, len(test_dataset)), 500)
            test_dataset = torch.utils.data.Subset(test_dataset, random_indices)
            dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    if dataset == 'mnist':
        data_root = MNIST_ROOT
        if split == 'train':
            train_dataset = torchvision.datasets.MNIST(data_root, download=False, transform=torchvision.transforms.ToTensor())
            random_indices = random.sample(range(0, len(train_dataset)), 500)
            train_dataset = torch.utils.data.Subset(train_dataset, random_indices)
            dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        elif split == 'test':
            test_dataset = torchvision.datasets.MNIST(data_root, train=False, download=False, transform=torchvision.transforms.ToTensor())
            random_indices = random.sample(range(0, len(test_dataset)), 500)
            test_dataset = torch.utils.data.Subset(test_dataset, random_indices)
            dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return dataloader


# -

def get_dataset(dataset: str, split: str, rotate: bool, loc: float, con: float,
        vm: bool) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split, rotate, loc, con, vm)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "mnist":
        return 10


# +
def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]


# -

def _cifar10(split: str, rotate: bool, loc: float, con: float, vm: bool) -> Dataset:
    if split == "train":
        if rotate:
            if vm:
                rotate = vonMisesRotate(loc, con)
            else:
                rotate = GaussianRotate(loc, con)
            tran = transforms.Compose([
		transforms.Pad(16,padding_mode='edge'),
		rotate,
		transforms.CenterCrop(32),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
            ])
        else:
            tran = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
            ])
        return datasets.CIFAR10(CIFAR10_DIR, train=True, download=True, transform=tran)

    elif split == "test":
        if rotate:
            if vm:
                rotate = vonMisesRotate(loc, con)
            else:
                rotate = GaussianRotate(loc, con)
            tran = transforms.Compose([
		transforms.Pad(16,padding_mode='edge'),
		rotatte,
		transforms.CenterCrop(32),
                        transforms.ToTensor()
                        ])
        else:
            tran = transforms.ToTensor()
        return datasets.CIFAR10(CIFAR10_DIR, train=False, download=True, transform=tran)


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        norm_input = (input[:,:num_channels//2, :,:] - means) / sds
        if num_channels > self.means.shape[0]:
            return torch.cat((norm_input, input[:, num_channels//2:, :, :]), dim=1)
        else:
            return norm_input
