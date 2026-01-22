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
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['subgroup'], f'The {BACKEND} backend does not support creating subgroups on CUDA devices')
@skip_if_lt_x_gpu(2)
def test_average_parameters(self):
    rank = dist.get_rank()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    device_id = rank_to_GPU[rank][0]
    model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.ReLU(), nn.Linear(1, 5, bias=False)).cuda(device_id)
    for p in model.parameters():
        p.data = torch.ones_like(p.data)
    model_averaging_utils.average_parameters(params=model.parameters(), process_group=None)
    for p in model.parameters():
        self.assertEqual(p.data, torch.ones_like(p.data))
    for p in model.parameters():
        p.data = torch.ones_like(p.data) * rank
    group_nccl = dist.new_group(ranks=[0, 1], backend='nccl')
    model_averaging_utils.average_parameters(params=model.parameters(), process_group=group_nccl)
    if not dist._rank_not_in_group(group_nccl):
        for p in model.parameters():
            self.assertEqual(p.data, torch.ones_like(p.data) * 0.5)
    else:
        for p in model.parameters():
            self.assertEqual(p.data, torch.ones_like(p.data) * rank)