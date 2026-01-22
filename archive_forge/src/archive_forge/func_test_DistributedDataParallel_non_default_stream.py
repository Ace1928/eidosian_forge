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
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
@skip_if_lt_x_gpu(int(os.environ['WORLD_SIZE']))
def test_DistributedDataParallel_non_default_stream(self):
    stream = torch.cuda.Stream(self.rank)
    rank = self.rank
    with torch.cuda.stream(stream):
        net = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False).cuda(rank), device_ids=[rank])
        for i in range(1000):
            grad = net.module.weight.grad
            if grad is not None:
                grad.requires_grad_(False)
                grad.zero_()
            batch = torch.tensor([rank]).float().cuda(rank)
            loss = net(batch).sum()
            loss.backward()
            grad = net.module.weight.grad
            avg = grad.clone()
            dist.all_reduce(avg)
            world_size = int(os.environ['WORLD_SIZE'])
            avg.div_(world_size)
            expected_grad = sum((i for i in range(world_size))) / world_size
            self.assertEqual(avg[0, 0], expected_grad, msg=f'Expected gradient of {expected_grad} but got {avg} on rank {self.rank}')