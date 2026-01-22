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
def check_for_expired_weak_storages(self):
    new_li = []
    stor_to_delete = []
    for obj in self.maybe_storages_to_delete:
        if not obj.expired():
            new_li.append(obj)
        else:
            stor_to_delete.append(obj)
    for obj in stor_to_delete:
        self.storage_memo.pop(obj, None)
    self.maybe_storages_to_delete = new_li
    self.check_expired_frequency = max(self.check_expired_frequency, len(self.maybe_storages_to_delete))