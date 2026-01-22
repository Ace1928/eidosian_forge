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
@skip_but_pass_in_sandcastle_if(BACKEND == 'nccl', 'Nccl does not support irecv')
def test_irecv(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        expected_tensors = [_build_tensor(src, -1) for src in range(1, world_size)]
        requests = [dist.irecv(expected_tensors[src - 1], src) for src in range(1, world_size)]
        for src in range(1, world_size):
            requests[src - 1].wait()
            self.assertTrue(requests[src - 1].is_completed())
            self.assertEqual(expected_tensors[src - 1], _build_tensor(src, 10))
    else:
        tensor = _build_tensor(rank, 10)
        dist.send(tensor, 0)
    self._barrier()