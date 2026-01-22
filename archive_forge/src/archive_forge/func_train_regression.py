import argparse
import os
import tempfile
from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
import ray
import ray.train as train
from ray.data import Dataset
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
def train_regression(num_workers=2, use_gpu=False):
    train_dataset, val_dataset = get_datasets()
    config = {'lr': 0.01, 'hidden_size': 20, 'batch_size': 4, 'epochs': 3}
    trainer = TorchTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu), datasets={'train': train_dataset, 'validation': val_dataset}, dataset_config=DataConfig(datasets_to_split=['train']))
    result = trainer.fit()
    print(result.metrics)
    return result