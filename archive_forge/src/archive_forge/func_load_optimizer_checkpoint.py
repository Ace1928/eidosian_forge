from pathlib import Path
from datetime import datetime
import torch
import time
from torch.distributed.fsdp import (
from torch.distributed._shard.checkpoint import (
from torch.distributed.checkpoint.default_planner import (
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist
def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """
    if not optimizer_checkpoint_path.is_file():
        print(f'warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. ')
        return
    full_osd = None
    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
    print(f'optimizer shard loaded on rank {rank}')