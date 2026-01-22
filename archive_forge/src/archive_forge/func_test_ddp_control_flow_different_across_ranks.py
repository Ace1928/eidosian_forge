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
@with_dist_debug_levels(levels=['OFF', 'INFO', 'DETAIL'])
@require_backend_is_available(DistTestCases.backend_feature['gpu'])
@skip_if_lt_x_gpu(2)
def test_ddp_control_flow_different_across_ranks(self):
    batch = 20
    dim = 10

    class ToyModel(nn.Module):

        def __init__(self, rank):
            super().__init__()
            self.lin1 = nn.Linear(10, 10, bias=False)
            self.lin2 = nn.Linear(10, 10, bias=False)
            self.rank = rank

        def forward(self, x):
            use_second_layer = torch.equal(x, torch.ones(batch, dim, device=x.device)) and self.rank == 1
            if use_second_layer:
                return self.lin2(F.relu(self.lin1(x)))
            else:
                return F.relu(self.lin1(x))
    world_size = dist.get_world_size()
    torch.cuda.set_device(self.rank)
    model = torch.nn.parallel.DistributedDataParallel(ToyModel(self.rank).cuda(self.rank), device_ids=[self.rank], find_unused_parameters=True)
    random_input = torch.randn(batch, dim, device=self.rank)
    ones_input = torch.ones(batch, dim, device=self.rank)
    for i in range(6):
        if i % 2 == 0:
            out = model(random_input)
        else:
            out = model(ones_input)
        loss = out.sum()
        loss.backward()
        local_used_map = model.reducer._get_local_used_map()
        if i % 2 == 0:
            expected = torch.tensor([world_size, 0], device=self.rank, dtype=torch.int32)
        else:
            expected = torch.tensor([world_size, 1], device=self.rank, dtype=torch.int32)
        variable_usage_tensor = local_used_map
        self.assertEqual(variable_usage_tensor, expected)
    model = torch.nn.parallel.DistributedDataParallel(ToyModel(self.rank).cuda(self.rank), device_ids=[self.rank], find_unused_parameters=False)
    for i in range(2):
        if i == 0:
            loss = model(random_input).sum()
            loss.backward()
        else:
            try:
                loss = model(random_input).sum()
                loss.backward()
            except RuntimeError as e:
                msg = str(e)
                verify_ddp_error_logged(model, msg)
                unused_param_index = 1
                expected_strs = [ddp_prev_reduction_unfinished_str, ddp_recommend_find_unused_params_str, ddp_outputs_not_used_in_loss_str, f'Parameter indices which did not receive grad for rank {self.rank}: {unused_param_index}']
                if dist.get_debug_level() == dist.DebugLevel.OFF:
                    expected_strs.append(ddp_suggest_debug_mode_str)
                else:
                    unreduced_params = ', '.join(['lin2.weight'])
                    expected_strs.append(f'did not receive grad for rank {self.rank}: {unreduced_params}')
                for s in expected_strs:
                    self.assertTrue(s in msg, f'Expected {s} to be in {msg}')
                self.assertFalse(ddp_find_unused_params_enabled_str in msg)
            else:
                self.assertFalse(True, 'DDP error not raised')
    dist.barrier()