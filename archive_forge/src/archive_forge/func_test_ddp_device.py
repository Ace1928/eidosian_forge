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
@require_backend_is_available(DistTestCases.backend_feature['gpu'])
@skip_if_lt_x_gpu(2)
def test_ddp_device(self):
    m = nn.Linear(10, 10).to(self.rank)
    expected_len = 2

    class TensorWrapper:
        __slots__ = ['t', 'moved_to_gpu']

        def __init__(self, t):
            self.t = t
            self.moved_to_gpu = False

    def tuple_and_list_validator(x):
        self.assertTrue(len(x), expected_len)
        self.assertEqual(1, len({t.device for t in x}))
        self.assertEqual(x[0].device.index, self.rank)
        return x[0] + x[1]

    def namedtuple_validator(x):
        self.assertEqual(x._fields, EXPECTED_FIELDS)
        self.assertEqual(x.a.device.index, x.b.device.index)
        self.assertEqual(x.a.device.index, self.rank)
        return x.a + x.b

    def custom_type_validator(x):
        self.assertTrue(x.moved_to_gpu or str(x.t.device) == 'cpu')
        x.t = x.t.to(self.rank)
        x.moved_to_gpu = True
        return x.t

    def dict_validator(x):
        self.assertTrue(EXPECTED_FIELDS[0] in x.keys())
        self.assertTrue(EXPECTED_FIELDS[1] in x.keys())
        self.assertEqual(1, len({t.device for t in x.values()}))
        self.assertEqual(x[EXPECTED_FIELDS[0]].device.index, self.rank)
        return x[EXPECTED_FIELDS[0]] + x[EXPECTED_FIELDS[1]]
    validators = {TensorWrapper: custom_type_validator, tuple: tuple_and_list_validator, list: tuple_and_list_validator, TestNamedTupleInput_0: namedtuple_validator, TestNamedTupleInput_1: namedtuple_validator, dict: dict_validator}

    class ToyModel(torch.nn.Module):

        def __init__(_self):
            super().__init__()
            _self.lin = nn.Linear(10, 10, bias=False)

        def forward(_self, x, expected_type):
            self.assertTrue(isinstance(x, expected_type))
            fwd_tensor = validators[expected_type](x)
            return _self.lin(fwd_tensor)
    model = torch.nn.parallel.DistributedDataParallel(ToyModel().to(self.rank), device_ids=[self.rank])

    def train_iter(inp, input_type):
        for _ in range(4):
            out = model(inp, input_type)
            out.sum().backward()
    inp = tuple((torch.randn(10, 10) for _ in range(expected_len)))
    train_iter(inp, tuple)
    inp = [torch.randn(10, 10) for _ in range(expected_len)]
    train_iter(inp, list)
    inp = TensorWrapper(torch.randn(10, 10))
    train_iter(inp, TensorWrapper)
    batch = 5
    dim = 10
    a = torch.rand(batch, dim)
    b = torch.rand(batch, dim)
    inp = TestNamedTupleInput_0(a, b)
    train_iter(inp, type(inp))
    inp = TestNamedTupleInput_1(a, b)
    train_iter(inp, type(inp))
    inp = {EXPECTED_FIELDS[0]: a, EXPECTED_FIELDS[1]: b}
    train_iter(inp, type(inp))