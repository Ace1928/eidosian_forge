import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def extend_acc_subgraph(self, tag: str):
    """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
    deps = self.find_reverse_deps(tag_id=int(tag.split('_')[-1]))
    self.update_reverse_deps_for_fusions(deps)
    parent_nodes = self.find_parent_nodes_of_subgraph(tag)
    visited_nodes: NodeSet = set()
    while parent_nodes:
        node = None
        for n in parent_nodes:
            if deps[n] <= visited_nodes and n in self.acc_nodes:
                node = n
                break
        if node is None:
            break
        node.tag = tag
        parent_nodes.remove(node)
        visited_nodes.add(node)
        if node in self.fusions:
            for fusion_node in self.fusions[node]:
                if fusion_node not in visited_nodes:
                    parent_nodes.add(fusion_node)
        for arg in node.all_input_nodes:
            if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                parent_nodes.add(arg)