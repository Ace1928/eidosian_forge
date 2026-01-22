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
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl', 'Only Nccl supports all_gather_v')
@skip_if_no_gpu
def test_all_gather_v_cuda(self):
    self._barrier()
    group, group_id, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    device_id = rank_to_GPU[rank][0]
    output_split_sizes = []
    for dst in group:
        output_split_sizes.append(dst + 1)
    sum_len = sum(output_split_sizes)
    value = 2
    for async_val in [True, False]:
        tensor = torch.empty(output_split_sizes[rank], sum_len, sum_len, dtype=torch.float).fill_(value).cuda(device_id)
        out_tensor = _build_tensor(sum_len, -1, device_id=device_id)
        req = dist.all_gather(list(torch.split(out_tensor, output_split_sizes)), tensor, group_id, async_val)
        if async_val:
            req.wait()
        expected_value = value
        expected_tensor = _build_tensor(sum_len, expected_value, device_id=device_id)
        self.assertEqual(out_tensor, expected_tensor)
    self._barrier()