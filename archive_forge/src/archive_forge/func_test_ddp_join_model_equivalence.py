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
def test_ddp_join_model_equivalence(self):
    batch = 3
    dim = 10
    learning_rate = 0.03
    model = nn.Linear(dim, dim, bias=False)
    inp = torch.rand(batch, dim, device=self.rank)
    local_model = copy.deepcopy(model)
    local_model = local_model.cuda(self.rank)
    rank_to_iter_mapping = {rank: 2 * (rank + 1) for rank in range(dist.get_world_size())}
    local_iters = sum(rank_to_iter_mapping.values())
    local_optim = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    for _ in range(local_iters):
        local_optim.zero_grad()
        out = local_model(inp)
        loss = out.sum()
        loss.backward()
        local_optim.step()
    num_iters = rank_to_iter_mapping[self.rank]
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(self.rank), device_ids=[self.rank])
    ddp_optim = torch.optim.SGD(model.parameters(), lr=learning_rate * dist.get_world_size())
    with net.join():
        for i in range(num_iters):
            ddp_optim.zero_grad()
            out = net(inp)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize(device=self.rank)
            ddp_optim.step()
    for (_, local_tensor), (_, dist_tensor) in zip(local_model.state_dict().items(), net.module.state_dict().items()):
        self.assertEqual(local_tensor, dist_tensor)