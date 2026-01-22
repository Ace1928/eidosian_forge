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
def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
    if 'val' in node.meta:
        fake = node.meta['val']
        if hasattr(fake, 'constant') and fake.constant is not None:
            return fake.constant
        return fake
    elif (tensor_meta := node.meta.get('tensor_meta')):
        assert self.fake_tensor_mode is not None
        return FakeTensor(self.fake_tensor_mode, torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device='meta', requires_grad=tensor_meta.requires_grad, memory_format=tensor_meta.memory_format), torch.device('cpu'))
    elif len(node.users) == 0:
        return None
    raise ExportPassBaseError(f'Cannot construct an input for graph module: {graph_module}.')