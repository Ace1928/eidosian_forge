from __future__ import annotations
import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional
import torch
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
from torch.utils import cpp_backtrace
def record_cpp_call_stack(self, frames_to_skip: int) -> infra.Stack:
    """Records the current C++ call stack in the diagnostic."""
    stack = _cpp_call_stack(frames_to_skip=frames_to_skip)
    stack.message = 'C++ call stack'
    self.with_stack(stack)
    return stack