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
def call_dist_op(self, profiling_title_postfix, is_async, op, *args, expect_event=True, secondary_op_call=None, profile_cuda=False, tensor_shapes=None, **kwargs):
    op_calls = [lambda: op(*args, **kwargs)]
    if secondary_op_call is not None:
        op_calls.append(secondary_op_call)
    autograd_profiler_ctx = torch.autograd.profiler.profile(use_cuda=profile_cuda, record_shapes=True)
    with autograd_profiler_ctx as prof:
        works = [op_call() for op_call in op_calls]
        if is_async:
            for work in works:
                work.wait()
    if expect_event and dist.get_backend() in PROFILING_SUPPORTED_BACKENDS:
        events = get_profiling_event(dist.get_backend() + profiling_title_postfix, autograd_profiler_ctx)
        if dist.get_debug_level() != dist.DebugLevel.DETAIL:
            self.assertEqual(len(events), len(op_calls))
        for e in events:
            self.assertTrue(e.is_async)
            self.assertEqual(e.count, 1)
            self.assertGreaterEqual(e.cpu_time, 0)
            if tensor_shapes is not None and dist.get_debug_level() != dist.DebugLevel.DETAIL:
                self.assertEqual(e.input_shapes, tensor_shapes, f'event shape: {e.input_shapes} vs tensor {tensor_shapes}')