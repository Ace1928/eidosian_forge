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
@skip_if_no_gpu
@require_backend_is_available(DistTestCases.backend_feature['gpu'])
def test_all_reduce_result_cuda(self):
    group, group_id, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    for src in group:
        if rank == src:
            tensor = _build_tensor(src + 1, 2)
        else:
            tensor = _build_tensor(src + 1, 10)
        tensor = tensor.cuda(rank_to_GPU[rank][0])
        opts = AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        if group_id == GroupMember.WORLD:
            work = _get_default_group().allreduce([tensor], opts)
        else:
            work = group_id.allreduce([tensor], opts)
        if BACKEND == 'gloo':
            try:
                with self.assertRaisesRegex(RuntimeError, 'Work needs to be completed before calling result'):
                    work.result()
            except AssertionError:
                self.assertTrue(work.is_completed())
            work.wait()
            result = work.result()
        else:
            result = work.result()
            work.wait()
        expected_value = 2 + 10 * (len(group) - 1)
        self.assertEqual(result, [_build_tensor(src + 1, expected_value)])
    self._barrier()