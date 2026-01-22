import abc
import contextlib
import weakref
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
def pack_to_cpu(tensor):
    if not pin_memory:
        return (tensor.device, tensor.cpu())
    packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout, pin_memory=device_module.is_available() and (not tensor.is_sparse))
    packed.copy_(tensor)
    return (tensor.device, packed)