from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
@_tensor_operation
def clock(self):
    raise NotImplementedError()