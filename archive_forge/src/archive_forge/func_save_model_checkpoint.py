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
def save_model_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """saving model via rank0 cpu streaming and full_state_dict"""
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()
        print(f'saving process: rank {rank}  done w model state_dict\n')
    if rank == 0:
        print(f'--> saving model ...')
        folder_name = cfg.dist_checkpoint_root_folder + '/' + cfg.dist_checkpoint_folder + '-' + cfg.model_name
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + '-' + str(epoch) + '.pt'
        save_full_path = str(save_dir) + '/' + save_name
        torch.save(cpu_state, save_full_path)
        print(f'model checkpoint saved for epoch {epoch} at {save_full_path}\n')