import argparse
from enum import Enum
import importlib
import logging
import tempfile
import time
from typing import Any, List, Optional, cast
from golden_configs import oss_mnist
import numpy as np
import torch
import torch.autograd.profiler as profiler
from torch.cuda.amp import GradScaler as TorchGradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from benchmarks.datasets.mnist import setup_cached_mnist
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim import OSS
from fairscale.optim.grad_scaler import ShardedGradScaler
def get_problem(rank, world_size, batch_size, device, model_name: str):
    logging.info(f'Using {model_name} for benchmarking')
    try:
        model = getattr(importlib.import_module('torchvision.models'), model_name)(pretrained=False).to(device)
    except AttributeError:
        model = getattr(importlib.import_module('timm.models'), model_name)(pretrained=False).to(device)

    def collate(inputs: List[Any]):
        return {'inputs': torch.stack([i[0] for i in inputs]).repeat(1, 3, 1, 1).to(device), 'label': torch.tensor([i[1] for i in inputs]).to(device)}
    transforms = []
    if model_name.startswith('vit'):
        pic_size = int(model_name.split('_')[-1])
        transforms.append(Resize(pic_size))
    transforms.append(ToTensor())
    dataset = MNIST(transform=Compose(transforms), download=False, root=TEMPDIR)
    sampler: Sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate)
    loss_fn = nn.CrossEntropyLoss()
    return (model, dataloader, loss_fn)