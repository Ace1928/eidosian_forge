import operator
import traceback
import typing
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from functorch.experimental.control_flow import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
    if not isinstance(x, FakeTensor) and isinstance(x, torch.Tensor):
        if x.is_quantized:
            x = torch.dequantize(x)
        try:
            assert self.fake_tensor_mode is not None
            _ = self.fake_tensor_mode.from_tensor(x)
            tensor_meta = None
        except UnsupportedFakeTensorException:
            tensor_meta = _extract_tensor_metadata(x)
        return tensor_meta
    else:
        return None