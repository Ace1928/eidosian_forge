from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap

    A decorator function to unwrap WrappedTensors and debugger_constexpr before calling the function.
    Can be combined with _infer_tensor decorator to harmonize args (everything to torch tensor).
    