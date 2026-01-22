import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def swap_node_to_partition(node, p0, p1, node_to_latency_mapping, transfer_rate_per_sec):
    """This function helps to swap one node from partition p0
            with all the nodes in another partition p1
            """
    p1_nodes = list(p1.nodes) + [None]
    min_cost = float('inf')
    node_pair: List[Node] = []
    for n1 in p1_nodes:
        if n1 is not None and n1.op in {'placeholder', 'get_attr'}:
            continue
        cost = try_swap_nodes(node, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec)
        if cost < min_cost:
            node_pair = [node, n1]
            min_cost = cost
    return (cost, node_pair)