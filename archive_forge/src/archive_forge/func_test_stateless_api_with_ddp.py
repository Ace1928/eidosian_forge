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
def test_stateless_api_with_ddp(self):

    class MockModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(1, 1)
            buffer = torch.ones(1)
            self.register_buffer('buffer', buffer)

        def forward(self, x):
            return self.l1(x) + self.buffer
    device = self.rank
    module = MockModule().to(device)
    module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device])
    x = torch.rand((1, 1)).to(device)
    weight = torch.tensor([[1.0]], device=device, requires_grad=True)
    bias = torch.tensor([0.0], device=device, requires_grad=True)
    buffer = torch.tensor([0.0], device=device)
    parameters = {'module.l1.weight': weight, 'module.l1.bias': bias, 'module.buffer': buffer}
    prev_weight = module.module.l1.weight.clone()
    prev_buffer = module.module.buffer.clone()
    res = torch.func.functional_call(module, parameters, x)
    self.assertEqual(x, res)
    cur_weight = module.module.l1.weight
    cur_buffer = module.module.buffer
    self.assertEqual(cur_weight, prev_weight)
    self.assertEqual(cur_buffer, prev_buffer)
    res.backward()
    self.assertIsNotNone(weight.grad)
    self.assertIsNotNone(bias.grad)
    self.assertIsNone(buffer.grad)
    self.assertIsNone(module.module.l1.weight.grad)
    self.assertIsNone(module.module.l1.bias.grad)
    self.assertIsNone(module.module.buffer.grad)