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
def test_ddp_shared_grad_acc_unused_params(self):

    class ToyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.net1 = nn.Linear(10, 5, bias=False)
            self.bias = nn.Parameter(torch.zeros(5))
            self.net1.bias = self.bias
            self.net2 = nn.Linear(10, 5)

        def forward(self, x):
            return self.net2(x).sum()
    torch.cuda.set_device(self.rank)
    model = ToyModel().to(torch.cuda.current_device())
    for static in [True, False]:
        ddp_model = torch.nn.parallel.DistributedDataParallel(copy.deepcopy(model), device_ids=[self.rank], find_unused_parameters=True, static_graph=static)
        inp = torch.randn(20, 10, device=self.rank)
        for i in range(6):
            loss = ddp_model(inp)
            loss /= 10
            loss.backward()