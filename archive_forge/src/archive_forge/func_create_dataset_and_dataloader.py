import argparse
import os
import math
import json
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import tqdm
import wandb
import numpy as np
from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.openchat_dataset import OpenchatDataset
def create_dataset_and_dataloader(args, epoch: int):
    filename = f'{args.data_prefix}.{epoch}.parquet'
    print(f'Loading epoch {epoch} data from {filename}...')
    dataset = OpenchatDataset(dataset_filename=filename, batch_max_length=args.batch_max_len, rank=dist.get_rank(), num_replicas=dist.get_world_size())
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=8, pin_memory=True)
    return (dataset, dataloader)