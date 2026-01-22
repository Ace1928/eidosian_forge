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
def call_cond(self, pred: ProxyValue, true_fn: torch.fx.GraphModule, false_fn: torch.fx.GraphModule, inputs: List[Argument], meta: NodeMetadata) -> ProxyValue:
    true_branch = self.call_submodule(true_fn, tuple(inputs))
    false_branch = self.call_submodule(false_fn, tuple(inputs))
    assert true_branch is not None
    assert false_branch is not None
    return self._fx('call_function', torch.ops.higher_order.cond, (pred, true_branch.graph_module, false_branch.graph_module, list(inputs)), {}, meta)