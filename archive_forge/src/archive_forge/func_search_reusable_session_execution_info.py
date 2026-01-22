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
def search_reusable_session_execution_info(self, graph_module: torch.fx.GraphModule, *args):
    if graph_module not in self.execution_info_per_graph_module:
        return None
    candidates = self.execution_info_per_graph_module[graph_module]
    for candidate in candidates:
        if candidate.is_supported(*args):
            return candidate
    return None