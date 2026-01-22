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
@property
def pg_coalesce_state(self) -> Dict[dist.ProcessGroup, List[Union[_CollOp, P2POp]]]:
    return self._get_world().pg_coalesce_state