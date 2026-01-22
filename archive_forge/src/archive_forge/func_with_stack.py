from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def with_stack(self: _Diagnostic, stack: infra.Stack) -> _Diagnostic:
    """Adds a stack to the diagnostic."""
    self.stacks.append(stack)
    return self