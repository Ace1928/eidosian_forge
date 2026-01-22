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
def test_ddp_static_graph_nested_types(self):
    rank = self.rank
    torch.cuda.set_device(rank)

    class NestedOutputModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(100, 1, bias=False)

        def forward(self, inp, output_type):
            if output_type == 'tuple':
                return (self.lin(inp), (self.lin(inp), self.lin(inp)))
            elif output_type == 'list':
                return [self.lin(inp), [self.lin(inp), self.lin(inp)]]
            elif output_type == 'dict':
                return {'a': self.lin(inp), 'b': {'c': self.lin(inp)}}

    def get_loss(model_output):
        loss = 0.0
        if isinstance(model_output, torch.Tensor):
            return model_output.sum()
        elif isinstance(model_output, dict):
            for value in model_output.values():
                loss += get_loss(value)
        elif isinstance(model_output, (tuple, list)):
            for x in model_output:
                loss += get_loss(x)
        else:
            raise ValueError(f'Unknown model output type {type(model_output)}')
        return loss
    model = NestedOutputModule().cuda(rank)
    model_static_graph = copy.deepcopy(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_static_graph = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    inp = torch.randn(10, 100)
    type_mapping = {'list': list, 'tuple': tuple, 'dict': dict}
    for output_type in type_mapping.keys():
        for i in range(6):
            out = model(inp, output_type=output_type)
            loss = get_loss(out)
            loss.backward()
            self._model_step(model)
            out_static = model_static_graph(inp, output_type=output_type)
            self.assertTrue(isinstance(out_static, type_mapping[output_type]))
            loss_static = get_loss(out_static)
            loss_static.backward()
            self._model_step(model_static_graph)
            for p, p_static in zip(model.parameters(), model_static_graph.parameters()):
                self.assertEqual(p, p_static)