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
def compute_tensor_metadata(self, t: torch.Tensor, h=None):
    if h is None:
        h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
    return (t.dtype, h, t.storage_offset(), tuple(t.shape), t.stride(), torch._utils.get_tensor_metadata(t))