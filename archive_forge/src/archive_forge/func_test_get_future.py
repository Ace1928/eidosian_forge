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
@skip_but_pass_in_sandcastle_if(BACKEND != 'mpi' and BACKEND != 'nccl' and (BACKEND != 'gloo'), 'get_future is only supported on mpi, nccl and gloo')
@nccl_skip_if_lt_x_gpu(BACKEND, 2)
def test_get_future(self):

    def mult(fut):
        return [t * 3 for t in fut.wait()]

    def add(fut):
        return [t + 1 for t in fut.wait()]
    group, group_id, rank = self._init_global_test()
    input = _build_tensor(3, 2)
    if BACKEND == 'nccl':
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        device_id = rank_to_GPU[rank][0]
        input = input.to(device_id)
    fut = group_id.allreduce([input]).get_future()
    res = fut.then(mult).then(add).wait()
    expected = _build_tensor(3, 2 * len(group) * 3 + 1)
    self.assertEqual(res[0], expected)