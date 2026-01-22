import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    verify_bfloat_support = torch.version.cuda and torch.cuda.is_bf16_supported() and (packaging.version.parse(torch.version.cuda).release >= (11, 0)) and dist.is_nccl_available() and (nccl.version() >= (2, 10)) or is_xpu_available()
    mixed_precision_policy = None
    wrapping_policy = None
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready and (not cfg.use_fp16):
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f'bFloat16 enabled for mixed precision - using bfSixteen policy')
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f'FP16 enabled')
        else:
            print(f'bFloat16 support not present. Using FP32, and not mixed precision')
    wrapping_policy = get_llama_wrapper()
    return (mixed_precision_policy, wrapping_policy)