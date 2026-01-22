from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import (
from torch.fx.passes.shape_prop import ShapeProp
@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    tensor_meta = node.meta.get('tensor_meta')
    if not tensor_meta:
        raise RuntimeError(f'Node {node} has no tensor metadata associated with it! Check that shape propagation has run.')
    return tensor_meta