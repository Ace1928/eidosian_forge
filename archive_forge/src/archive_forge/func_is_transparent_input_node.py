from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def is_transparent_input_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
    if node.op == 'placeholder' or node not in partition or node in removed_nodes:
        return True
    if node in transparent_input_nodes:
        return transparent_input_nodes[node]
    if is_non_compute_node(node):
        for input_n in node.all_input_nodes:
            if not is_transparent_input_node(input_n, partition, removed_nodes):
                transparent_input_nodes[node] = False
                return False
        transparent_input_nodes[node] = True
        return True
    transparent_input_nodes[node] = False
    return False