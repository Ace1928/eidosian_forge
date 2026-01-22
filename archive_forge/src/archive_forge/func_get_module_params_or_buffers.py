import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def get_module_params_or_buffers():
    for pname, ptensor in chain(leaf_module.named_parameters(), leaf_module.named_buffers()):
        pname1 = node.name + '.' + pname
        label1 = pname1 + '|op_code=get_' + 'parameter' if isinstance(ptensor, torch.nn.Parameter) else 'buffer' + '\\l'
        dot_w_node = pydot.Node(pname1, label='{' + label1 + self._get_tensor_label(ptensor) + '}', **_WEIGHT_TEMPLATE)
        dot_graph.add_node(dot_w_node)
        dot_graph.add_edge(pydot.Edge(pname1, node.name))