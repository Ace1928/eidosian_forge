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
def read_tensor_metadata(self, name: str):
    fn = os.path.join(self.loc, 'tensors', name)
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    return torch.load(fn, weights_only=True)