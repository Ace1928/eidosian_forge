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
def test_rref_context_debug_info(self):
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    rref1 = RRef(self.rank)
    info = _rref_context_get_debug_info()
    self.assertIn('num_owner_rrefs', info)
    self.assertIn('num_pending_users', info)
    self.assertEqual(0, int(info['num_owner_rrefs']))
    self.assertEqual(0, int(info['num_pending_users']))
    dist.barrier()
    dst_rank = (self.rank + 1) % self.world_size
    rpc.rpc_sync(worker_name(dst_rank), set_global_rref, args=(rref1,))
    wait_until_pending_futures_and_users_flushed()
    dist.barrier()
    info = _rref_context_get_debug_info()
    self.assertIn('num_owner_rrefs', info)
    self.assertEqual(1, int(info['num_owner_rrefs']))
    self.assertEqual(0, int(info['num_pending_users']))
    dist.barrier()
    rpc.rpc_sync(worker_name(dst_rank), clear_global_rref)
    while int(info['num_owner_rrefs']) != 0:
        info = _rref_context_get_debug_info()
        time.sleep(0.1)
    dist.barrier()
    rref2 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
    rref3 = rpc.remote(worker_name(dst_rank), torch.add, args=(torch.ones(2, 2), 1))
    rref2.to_here()
    rref3.to_here()
    wait_until_pending_futures_and_users_flushed()
    dist.barrier()
    info = _rref_context_get_debug_info()
    self.assertIn('num_owner_rrefs', info)
    self.assertEqual(2, int(info['num_owner_rrefs']))
    self.assertEqual(0, int(info['num_pending_users']))
    dist.barrier()