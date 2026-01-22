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
def create_torch_iterator(shard):
    iterator = shard.iter_torch_batches(batch_size=batch_size)
    for batch in iterator:
        yield (batch['x'].float(), batch['y'].float())