import os
import argparse
from filelock import FileLock
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
def get_data_loaders(batch_size=64):
    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    with FileLock(os.path.expanduser('~/data.lock')):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=True, download=True, transform=mnist_transforms), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('~/data', train=False, download=True, transform=mnist_transforms), batch_size=batch_size, shuffle=True)
    return (train_loader, test_loader)