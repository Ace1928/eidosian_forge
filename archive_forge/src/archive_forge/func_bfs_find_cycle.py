import copy
from queue import SimpleQueue
from typing import List, Dict, Tuple
import torch.fx
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph
from torch.fx.passes.utils import lift_subgraph_as_module
from torch.fx._compatibility import compatibility
def bfs_find_cycle(root_nodes: NodeList) -> bool:
    visited: NodeSet = set()
    queue: NodeList = root_nodes
    while queue:
        current = queue.pop()
        visited.add(current)
        if current in partition_set:
            return True
        for user_node in current.users:
            if user_node in visited:
                continue
            queue.append(user_node)
    return False