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
def train_mnist(config):
    should_checkpoint = config.get('should_checkpoint', False)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    while True:
        train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)
        metrics = {'mean_accuracy': acc}
        if should_checkpoint:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(model.state_dict(), os.path.join(tempdir, 'model.pt'))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)