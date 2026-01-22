import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl', 'Only Nccl supports reduce_scatter_v')
@skip_but_pass_in_sandcastle_if(BACKEND in DistTestCases.skip_collective['reduce'], f'{BACKEND} does not support reduce')
@skip_if_no_gpu
def test_reduce_scatter_v_cuda(self):
    self._barrier()
    group, group_id, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    device_id = rank_to_GPU[rank][0]
    input_split_sizes = []
    for src in group:
        input_split_sizes.append(src + 1)
    start_len = sum(input_split_sizes[:rank])
    end_len = start_len + input_split_sizes[rank]
    sum_len = sum(input_split_sizes)
    master_value = 2
    worker_value = 10
    for async_val in [True, False]:
        tensor = _build_tensor(sum_len, worker_value, device_id=device_id)
        tensor[start_len:end_len].fill_(master_value)
        out_tensor = torch.empty(input_split_sizes[rank], sum_len, sum_len, dtype=torch.float).fill_(-1).cuda(device_id)
        req = dist.reduce_scatter(out_tensor, list(torch.split(tensor, input_split_sizes)), dist.ReduceOp.SUM, group_id, async_val)
        if async_val:
            req.wait()
        expected_value = 2 + 10 * (len(group) - 1)
        expected_tensor = torch.empty(input_split_sizes[rank], sum_len, sum_len, dtype=torch.float)
        expected_tensor = expected_tensor.fill_(expected_value).cuda(device_id)
        self.assertEqual(out_tensor, expected_tensor)
    self._barrier()