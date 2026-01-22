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
@require_world_size(2)
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
def test_ddp_compile_static_graph(self):
    """Tests that DDP works with torch compile when static_graph=True"""
    model = torch.nn.Linear(10, 10).cuda(self.rank)
    model_clone = copy.deepcopy(model)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
    ddp_static = torch.nn.parallel.DistributedDataParallel(model_clone, device_ids=[self.rank], static_graph=True)
    ddp = torch.compile(ddp)
    ddp_static = torch.compile(ddp_static)
    input = torch.rand(10, 10).cuda(self.rank)
    for _ in range(6):
        out_ddp = ddp(input).sum()
        out_ddp_static = ddp_static(input).sum()
        self.assertEqual(out_ddp, out_ddp_static)
        out_ddp.backward()
        out_ddp_static.backward()
        for p1, p2 in zip(ddp.parameters(), ddp_static.parameters()):
            self.assertEqual(p1.grad, p2.grad)