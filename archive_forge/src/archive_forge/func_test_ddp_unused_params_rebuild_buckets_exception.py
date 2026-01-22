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
def test_ddp_unused_params_rebuild_buckets_exception(self):

    class ToyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.net1 = nn.Linear(10, 10, bias=False)
            self.net2 = nn.Linear(10, 10, bias=False)

        def forward(self, x):
            return self.net1(x)
    ddp = torch.nn.parallel.DistributedDataParallel(ToyModel().cuda(self.rank), device_ids=[self.rank])
    for i in range(2):
        inp = torch.rand(1, 10)
        if i > 0:
            try:
                ddp(inp).sum().backward()
            except RuntimeError as e:
                msg = str(e)
                verify_ddp_error_logged(ddp, msg)
                expected_strs = [ddp_prev_reduction_unfinished_str, ddp_recommend_find_unused_params_str, ddp_outputs_not_used_in_loss_str]
                if dist.get_debug_level() == dist.DebugLevel.OFF:
                    expected_strs.append(ddp_suggest_debug_mode_str)
                else:
                    unreduced_params = ', '.join(['net2.weight'])
                    expected_strs.append(f'did not receive grad for rank {self.rank}: {unreduced_params}')
                for s in expected_strs:
                    self.assertTrue(s in msg, f'Expected {s} to be in {msg}')
                self.assertFalse(ddp_find_unused_params_enabled_str in msg)
            else:
                self.assertFalse(True, 'DDP unused parameters error not raised.')
        else:
            ddp(inp).sum().backward()
    dist.barrier()