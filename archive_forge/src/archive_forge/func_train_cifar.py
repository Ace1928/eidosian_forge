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
def train_cifar(config):
    net = Net(config['l1'], config['l2'])
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    if config['smoke_test']:
        trainset, testset = load_test_data()
    else:
        trainset, testset = load_data(DATA_DIR)
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=0 if config['smoke_test'] else 8)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=0 if config['smoke_test'] else 8)
    for epoch in range(10):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = (inputs.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / epoch_steps))
                running_loss = 0.0
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = (inputs.to(device), labels.to(device))
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({'loss': val_loss / val_steps, 'accuracy': correct / total}, checkpoint=checkpoint)
    print('Finished Training')