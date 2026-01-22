import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
def write_tensor(self, name: str, t: torch.Tensor) -> None:
    storage = t.untyped_storage()
    h = self.write_storage(storage)
    d, f = os.path.split(name)
    payload = self.compute_tensor_metadata(t, h=h)
    subfolder = os.path.join(self.loc, 'tensors', d)
    os.makedirs(subfolder, exist_ok=True)
    torch.save(payload, os.path.join(subfolder, f))