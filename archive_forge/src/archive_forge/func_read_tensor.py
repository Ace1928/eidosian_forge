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
def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
    dtype, h, storage_offset, size, stride, metadata = self.read_tensor_metadata(name)
    storage = self.read_storage(h, device=device)
    t = torch.tensor([], dtype=dtype, device=storage.device)
    t.set_(storage, storage_offset, size, stride)
    torch._utils.set_tensor_metadata(t, metadata)
    return t