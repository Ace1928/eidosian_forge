import argparse
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import ray.train as train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
y = a * x + b