from __future__ import annotations
import contextlib
import copy
import inspect
import io
import re
import textwrap
import typing
import warnings
from typing import (
import torch
import torch._C._onnx as _C_onnx
import torch.jit._trace
import torch.serialization
from torch import _C
from torch.onnx import (  # noqa: F401
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import (
@contextlib.contextmanager
@_beartype.beartype
def setup_onnx_logging(verbose: bool):
    is_originally_enabled = torch.onnx.is_onnx_log_enabled()
    if is_originally_enabled or verbose:
        torch.onnx.enable_log()
    try:
        yield
    finally:
        if not is_originally_enabled:
            torch.onnx.disable_log()