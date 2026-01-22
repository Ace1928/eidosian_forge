from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
def recursive_add_node(self, fusion_group: 'FxNetAccFusionsFinder.FusionGroup', inputs: Union[NodeSet, NodeList]):
    """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
    for arg in inputs:
        if arg.op not in CALLABLE_NODE_OPS:
            continue
        if self.nodes.index(arg) < fusion_group.top_node_idx:
            continue
        if arg in fusion_group.nodes:
            return True
        if self.recursive_add_node(fusion_group, arg.all_input_nodes):
            fusion_group.add_node(arg)
            return True
    return False