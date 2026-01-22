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
@skip_if_lt_x_gpu(2)
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
@unittest.skip('Test is failing, tracking issue at https://github.com/pytorch/pytorch/issues/102751')
def test_ddp_has_finalized(self):

    @dataclass
    class MyClass:
        obj: torch.Tensor

    class MyModel(nn.Module):

        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            self.fc1 = nn.Linear(1024, 1024).cuda(rank)
            self.fc2 = nn.Linear(1024, 2 * 1024).cuda(rank)

        def forward(self, inp):
            if self.rank == 0:
                return (self.fc1(inp), MyClass(self.fc2(inp)))
            else:
                return (self.fc1(inp), self.fc2(inp))
    model = MyModel(self.rank)
    input = torch.rand(10, 1024, requires_grad=True).cuda(self.rank)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], find_unused_parameters=True, bucket_cap_mb=1024 * 4 / 1024 / 1024)
    if self.rank == 0:
        out1, _ = ddp(input)
        out1.sum().backward()
    else:
        out1, out2 = ddp(input)
        (out1.sum() + out2.sum()).backward()
    if self.rank == 0:
        with self.assertRaisesRegex(RuntimeError, 'Expected to have finished reduction in the prior iteration'):
            ddp._check_reducer_finalized()
        with self.assertRaisesRegex(RuntimeError, 'Expected to have finished reduction in the prior iteration'):
            ddp(input)
    else:
        ddp._check_reducer_finalized()
        ddp(input)