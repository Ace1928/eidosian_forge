import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def finish_mpu_init():
    args = get_args()
    device_count = torch.cuda.device_count()
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    if device_count > 0:
        device = args.rank % device_count
        if args.local_rank is not None:
            assert args.local_rank == device, 'expected local-rank to be the same as rank % device-count.'
        else:
            args.local_rank = device
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.virtual_pipeline_model_parallel_size, args.pipeline_model_parallel_split_rank)
    if args.rank == 0:
        print(f'> setting random seeds to {args.seed} ...')
    _set_random_seed(args.seed, args.data_parallel_random_init)