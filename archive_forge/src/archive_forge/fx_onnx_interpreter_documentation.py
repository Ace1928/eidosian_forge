from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
Export a fx.GraphModule submodule to ONNXScript graph.

        The export process specifically targets `call_module` nodes that are created by
        the exporter's `Modularize` pass. Each `call_module` node has an associated fx.GraphModule
        by `node.target` underneath the root fx.GraphModule. These `call_module` nodes are exported as ONNX
        function nodes. The related `sub_module` is then exported as an ONNX model local function,
        which is represented by another `TorchScriptGraph`. This `TorchScriptGraph` sets the current
        `onnxscript_graph` as its parent.

        Args:
            node: The call_module node in the FX graph that represents the submodule call.
            parent_onnxscript_graph: The parent ONNXScript graph to which the ONNX function and
                function node belong.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNXScript value.
            tracer: The tracer used to trace the ONNXScript graph.
            root_fx_graph_module: The root FX module.
            onnxfunction_dispatcher: The dispatcher.
            op_level_debug: Whether to enable op-level debug.
        