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
def test_custom_exception_throw_during_reconstruction(self):
    """
        Test that we still throw info about the remote side exception even when
        we cannot recreate it on client side.
        """
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    if self.rank != 0:
        exc_caught = False
        dst = worker_name(0)
        try:
            rpc.rpc_sync(dst, custom_raise_func, args=())
        except RuntimeError as e:
            exc_caught = True
            msg = str(e)
            print(f'Got msg {msg}')
            self.assertTrue('Original exception on remote side was' in msg)
            self.assertTrue('CustomException' in msg)
        except BaseException as e:
            raise RuntimeError(f'Failure - expected RuntimeError, got {e}') from e
        finally:
            self.assertTrue(exc_caught)
    dist.barrier()