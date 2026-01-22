from __future__ import annotations
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra
Generates a FX GraphModule using torch.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    