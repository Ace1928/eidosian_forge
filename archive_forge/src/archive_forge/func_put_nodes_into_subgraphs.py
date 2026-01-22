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
def put_nodes_into_subgraphs(self) -> List[Subgraph]:
    current_cpu_nodes, current_acc_nodes = self.starter_nodes()
    visited_nodes: NodeSet = set()
    acc_subgraph: bool = not any((len(self.deps[n]) == 0 for n in current_cpu_nodes))
    current_subgraph_nodes: NodeList = []
    subgraphs: List[Subgraph] = []
    while current_cpu_nodes or current_acc_nodes:
        current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
        node = next((n for n in current_nodes if self.deps[n] <= visited_nodes), None)
        if node is None:
            if not current_subgraph_nodes:
                raise FxNetSplitterInternalError("Subgraph can't be empty")
            subgraphs.append(Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes))
            acc_subgraph = not acc_subgraph
            current_subgraph_nodes = []
            continue
        current_nodes.remove(node)
        visited_nodes.add(node)
        current_subgraph_nodes.append(node)
        if node in self.fusions:
            if node in self.acc_nodes:
                current_acc_nodes.update(self.fusions[node] - visited_nodes)
            else:
                current_cpu_nodes.update(self.fusions[node] - visited_nodes)
        for user in node.users:
            if user.op not in CALLABLE_NODE_OPS:
                continue
            if user in self.acc_nodes:
                current_acc_nodes.add(user)
            else:
                current_cpu_nodes.add(user)
    if current_subgraph_nodes:
        subgraphs.append(Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes))
    if not subgraphs:
        raise FxNetSplitterInternalError("Couldn't create subgraphs")
    return subgraphs