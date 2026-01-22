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
def create_model(args):
    print(f'Loading model {args.model_type} from {args.model_path}...')
    model = MODEL_CONFIG_MAP[args.model_type].model_create_for_training(args.model_path)
    model = model.to(args.local_rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps, fused=True)
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(), optimizer=optimizer)
    args.device = model_engine.device
    return (model_engine, optimizer)