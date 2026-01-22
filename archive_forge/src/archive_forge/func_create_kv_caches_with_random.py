import enum
import os
import socket
import subprocess
import uuid
from platform import uname
from typing import List, Tuple, Union
from packaging.version import parse, Version
import psutil
import torch
import asyncio
from functools import partial
from typing import (
from collections import OrderedDict
from typing import Any, Hashable, Optional
from vllm.logger import init_logger
def create_kv_caches_with_random(num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_size: int, cache_dtype: Optional[Union[str, torch.dtype]], model_dtype: Optional[Union[str, torch.dtype]]=None, seed: Optional[int]=0, device: Optional[str]='cuda') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if isinstance(cache_dtype, str):
        if cache_dtype == 'auto':
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f'Invalid model dtype: {model_dtype}')
        elif cache_dtype in ['half', 'bfloat16', 'float']:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == 'fp8_e5m2':
            torch_dtype = torch.uint8
        else:
            raise ValueError(f'Invalid kv cache dtype: {cache_dtype}')
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f'Invalid kv cache dtype: {cache_dtype}')
    scale = head_size ** (-0.5)
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype == 'fp8_e5m2':
            _generate_random_fp8_e5m2(key_cache, -scale, scale)
        elif torch_dtype in [torch.half, torch.bfloat16, torch.float]:
            key_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f'Does not support key cache of type {cache_dtype}')
        key_caches.append(key_cache)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype == 'fp8_e5m2':
            _generate_random_fp8_e5m2(value_cache, -scale, scale)
        elif torch_dtype in [torch.half, torch.bfloat16, torch.float]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(f'Does not support value cache of type {cache_dtype}')
        value_caches.append(value_cache)
    return (key_caches, value_caches)