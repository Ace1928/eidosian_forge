import os
from typing import Dict
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
def train_fashion_mnist(num_workers=2, use_gpu=False):
    global_batch_size = 32
    train_config = {'lr': 0.001, 'epochs': 10, 'batch_size_per_worker': global_batch_size // num_workers}
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TorchTrainer(train_loop_per_worker=train_func_per_worker, train_loop_config=train_config, scaling_config=scaling_config)
    result = trainer.fit()
    print(f'Training result: {result}')