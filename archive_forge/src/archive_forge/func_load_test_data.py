import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
def load_test_data():
    trainset = torchvision.datasets.FakeData(128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor())
    testset = torchvision.datasets.FakeData(16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor())
    return (trainset, testset)