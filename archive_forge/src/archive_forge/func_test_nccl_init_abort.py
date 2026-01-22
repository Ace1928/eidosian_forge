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
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl', 'TORCH_NCCL_USE_COMM_NONBLOCKING only applies to NCCL')
def test_nccl_init_abort(self):
    """
            Tests that we can abort a NCCL communicator during initialization and
            recover appropriately.
            """
    os.environ['TORCH_NCCL_USE_COMM_NONBLOCKING'] = '1'
    dist.destroy_process_group()
    timeout = timedelta(seconds=1)
    dist.init_process_group(init_method=INIT_METHOD, backend=BACKEND, world_size=int(os.environ['WORLD_SIZE']), rank=self.rank, timeout=timeout)
    running = True

    def abort(device):
        pg = _get_default_group()
        while running:
            pg._get_backend(torch.device(device))._shutdown()
            time.sleep(1)
    if self.rank != 1:
        import threading
        t = threading.Thread(target=abort, args=(self.rank,))
        t.start()
        with self.assertRaises(RuntimeError):
            torch.distributed.barrier()
        running = False
        t.join()