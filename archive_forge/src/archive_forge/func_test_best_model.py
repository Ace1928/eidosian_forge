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
def test_best_model(config: Dict, checkpoint: 'Checkpoint', smoke_test=False):
    best_trained_model = Net(config['l1'], config['l2'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_trained_model.to(device)
    with checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        model_state, optimizer_state = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state)
    if smoke_test:
        _, testset = load_test_data()
    else:
        _, testset = load_data(DATA_DIR)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = (images.to(device), labels.to(device))
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Best trial test set accuracy: {}'.format(correct / total))