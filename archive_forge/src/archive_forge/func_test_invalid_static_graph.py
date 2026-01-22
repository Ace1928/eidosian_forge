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
def test_invalid_static_graph(self):
    world_size = dist.get_world_size()
    torch.cuda.set_device(self.rank)
    model = torch.nn.parallel.DistributedDataParallel(ControlFlowToyModel().cuda(self.rank), device_ids=[self.rank], static_graph=True)
    random_input = torch.randn(20, 10, device=self.rank)
    ones_input = torch.ones(20, 10, device=self.rank)
    expected_err = 'Your training graph has changed in this iteration'
    with self.assertRaisesRegex(RuntimeError, expected_err):
        for i in range(2):
            if i % 2 == 0:
                out = model(random_input)
            else:
                out = model(ones_input)
            loss = out.sum()
            loss.backward()
    verify_ddp_error_logged(model, expected_err)
    with self.assertRaisesRegex(RuntimeError, 'Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your training graph has changed in this iteration, e.g., one parameter is used in first iteration, but then got unused in the second iteration. this is not compatible with static_graph set to True.\nParameter indices which did not receive grad for'):
        for i in range(2):
            if i % 2 != 0:
                out = model(random_input)
            else:
                out = model(ones_input)
            loss = out.sum()
            loss.backward()
    verify_ddp_error_logged(model, 'Expected to have finished reduction')