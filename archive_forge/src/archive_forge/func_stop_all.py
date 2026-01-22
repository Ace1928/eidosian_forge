import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from ray.tune.examples.mnist_pytorch import train_func, test_func, ConvNet,\
import ray
from ray import train, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.utils import validate_save_restore
def stop_all(self):
    return self.should_stop