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
def check_same_models_across_ranks(model: torch.nn.Module, process_group: Any, params_should_be_equal: bool, check_broadcast_buffers: bool) -> None:
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    for param in model.parameters():
        receptacle = [param.clone() for _ in range(world_size)]
        dist.all_gather(receptacle, param, group=process_group)
        if rank == 0:
            for sync_p in receptacle[1:]:
                assert not params_should_be_equal or torch.all(torch.eq(receptacle[0], sync_p)), f'Models differ in between ranks {receptacle[0]} - {sync_p}'
    if check_broadcast_buffers:
        for buffer in model.buffers():
            receptacle = [buffer.clone() for _ in range(world_size)]
            dist.all_gather(receptacle, buffer, group=process_group)
            if rank == 0:
                for sync_b in receptacle[1:]:
                    assert not params_should_be_equal or torch.all(torch.eq(receptacle[0], sync_b)), f'Models differ in between ranks {receptacle[0]} - {sync_b}'