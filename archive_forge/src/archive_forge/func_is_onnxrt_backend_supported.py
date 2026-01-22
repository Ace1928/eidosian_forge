import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def is_onnxrt_backend_supported() -> bool:
    """Returns ``True`` if ONNX Runtime dependencies are installed and usable
    to support TorchDynamo backend integration; ``False`` otherwise.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> if torch.onnx.is_onnxrt_backend_supported():
        ...     @torch.compile(backend="onnxrt")
        ...     def f(x):
        ...             return x * x
        ...     print(f(torch.randn(10)))
        ... else:
        ...     print("pip install onnx onnxscript onnxruntime")
        ...
    """
    return _SUPPORT_ONNXRT