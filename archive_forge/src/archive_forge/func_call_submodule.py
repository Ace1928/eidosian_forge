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
def call_submodule(self, graph_module: fx.GraphModule, inputs: Tuple[Argument, ...]) -> PassResult:
    prev_tracer, self.tracer = (self.tracer, self.ExportTracer(self, graph_module.graph._codegen))
    self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
    interpreter = self.ExportInterpreter(self, graph_module)
    prev_interpreter, self.interpreter = (self.interpreter, torch.fx.Interpreter(torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())))
    inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
    with fx_traceback.preserve_node_meta():
        interpreter.run(*inputs_data)
    new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)
    self.tracer = prev_tracer
    self.interpreter = prev_interpreter
    return PassResult(new_graph_module, True)