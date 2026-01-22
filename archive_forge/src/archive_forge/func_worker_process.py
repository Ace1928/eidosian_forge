import contextlib
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import random
from statistics import mean
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy
import pytest
import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import rpc
import torch.multiprocessing as mp
import torch.nn as nn
from fairscale.internal import torch_version
from fairscale.nn.model_parallel import destroy_model_parallel, initialize_model_parallel
from fairscale.nn.model_parallel.random import model_parallel_cuda_manual_seed
def worker_process(rank: int, world_size: int, filename: str, filename_rpc: str, func: Callable, args: Any, error_queue: Any) -> None:
    """Main function for unit tests launched with torch_spawn"""
    if not dist_init(rank, world_size, filename, filename_rpc):
        logging.warning('failed initializing torch distributed')
        teardown()
        return
    kwargs = {}
    if 'OMPI_COMM_WORLD_RANK' not in os.environ:
        kwargs['pipeline_backend'] = 'gloo'
    initialize_model_parallel(1, world_size, **kwargs)
    context = torch.backends.cudnn.flags(benchmark=False, deterministic=True) if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'flags') else contextlib.suppress()
    if torch.cuda.is_available() and (not hasattr(torch.backends.cudnn, 'flags')):
        make_cudnn_deterministic()
    try:
        with context:
            func(*args)
        teardown()
    except BaseException as e:
        logging.warning(f' Rank {rank}: {e}')
        teardown()
        if e.__class__.__name__ == 'Skipped':
            error_queue.put(str(e))
            return
        raise e