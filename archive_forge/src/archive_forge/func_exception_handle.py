import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
@classmethod
def exception_handle(cls, exc):
    cls._terminate.set()
    for coll in cls._cur_coll_on_pgs.values():
        with coll._start_cond:
            coll._start_cond.notify()
        with coll._done_cond:
            coll._done_cond.notify_all()