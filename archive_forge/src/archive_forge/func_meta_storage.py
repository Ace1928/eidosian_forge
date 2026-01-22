import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def meta_storage(self, s, callback):
    swr = StorageWeakRef(s)
    if swr not in self.storage_memo:
        self.storage_memo[swr] = callback(lambda: torch.empty(s.size(), dtype=torch.uint8, device='meta')).untyped_storage()
    return self.storage_memo[swr]