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
def test_get_worker_infos(self):
    worker_infos = rpc.api._get_current_rpc_agent().get_worker_infos()
    worker_names = {worker_info.name for worker_info in worker_infos}
    expected_worker_names = {worker_name(rank) for rank in range(self.world_size)}
    self.assertEqual(worker_names, expected_worker_names)
    worker_ids = {worker_info.id for worker_info in worker_infos}
    expected_worker_ids = set(range(self.world_size))
    self.assertEqual(worker_ids, expected_worker_ids)