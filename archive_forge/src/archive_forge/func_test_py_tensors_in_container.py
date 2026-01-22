import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
@dist_init
def test_py_tensors_in_container(self):
    n = self.rank + 1
    dst_rank = n % self.world_size
    a = [torch.ones(n, n), torch.ones(n, n)]
    b = TensorClass(build_complex_tensors())
    c = {'foo': torch.ones(n, n), 'bar': torch.ones(n, n)}
    ret = rpc.rpc_sync(worker_name(dst_rank), my_complex_tensor_function, args=(a, b, c))
    self.assertEqual(ret, my_complex_tensor_function(a, b, c))