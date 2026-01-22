from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def record_python_call_stack(self, frames_to_skip: int) -> infra.Stack:
    """Records the current Python call stack."""
    frames_to_skip += 1
    stack = utils.python_call_stack(frames_to_skip=frames_to_skip)
    self.with_stack(stack)
    if len(stack.frames) > 0:
        self.with_location(stack.frames[0].location)
    return stack