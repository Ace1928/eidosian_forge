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
def test_periodic_model_averager_param_group(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
    device_id = rank_to_GPU[rank][0]
    model = nn.Linear(1, 5, bias=False).cuda(device_id)
    param = next(model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    period = 4
    for warmup_steps in [12, 13, 14, 15]:
        averager = averagers.PeriodicModelAverager(period=period, warmup_steps=warmup_steps)
        for step in range(0, 20):
            for param_group in opt.param_groups:
                for params in param_group['params']:
                    params.grad = torch.ones_like(param.data) * rank
                    params.data = torch.ones_like(param.data) * rank
            averager.average_parameters(opt.param_groups)
            if step >= warmup_steps and (step - warmup_steps) % period == 0:
                for param_group in opt.param_groups:
                    for params in param_group['params']:
                        if params.grad is None:
                            continue
                        self.assertEqual(param.data, torch.ones_like(param.data) * sum(range(world_size)) / world_size)
            else:
                for param_group in opt.param_groups:
                    for params in param_group['params']:
                        if params.grad is None:
                            continue
                        self.assertEqual(param.data, torch.ones_like(param.data) * rank)