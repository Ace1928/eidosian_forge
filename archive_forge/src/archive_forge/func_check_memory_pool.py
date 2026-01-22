important optimization when chaining multiple CUDA graphs together, as it
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import logging
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from enum import auto, Enum
from typing import (
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.compile_fx import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.types import _bool
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef
from . import config
def check_memory_pool(device, pool_id, live_storages_ptrs: List[StorageWeakRefWrapper]):
    assert all((isinstance(elem, StorageWeakRefWrapper) for elem in live_storages_ptrs))
    unique_storages = {stor.data_ptr() for stor in live_storages_ptrs if stor()}
    if torch._C._cuda_checkPoolLiveAllocations(device, pool_id, unique_storages):
        return
    gc.collect()
    segments = get_cudagraph_segments(pool_id)
    allocated_not_in_live_storages = {}
    for segment in segments:
        addr = segment['address']
        for block in segment['blocks']:
            if block['state'] == 'active_allocated':
                if addr not in unique_storages:
                    allocated_not_in_live_storages[addr] = block
                else:
                    unique_storages.remove(addr)
            addr += block['size']
    torch._check(len(unique_storages) == 0, lambda: f'These storage data ptrs are not allocated in pool {pool_id} but should be {unique_storages}')
    if allocated_not_in_live_storages != 0:
        formatted = []
        for dp, block in allocated_not_in_live_storages.items():
            trace = format_tb(block.get('frames', []))
            formatted.append(f'Data Pointer: {dp}, history: \n{trace}')
        formatted_s = '\n'.join(formatted)
        msg = f'These live storage data ptrs are in the cudagraph pool but not accounted for as an output of cudagraph trees: \n\n{formatted_s}'
        raise RuntimeError(msg)