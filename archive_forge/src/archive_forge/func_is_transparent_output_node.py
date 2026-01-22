from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def is_transparent_output_node(node: Node, partition: Set[Node], removed_nodes: Set[Node]):
    if node.op == 'placeholder' or node not in partition or node in removed_nodes:
        return True
    if node in transparent_output_nodes:
        return transparent_output_nodes[node]
    if is_non_compute_node(node):
        for output_n in node.users:
            if not is_transparent_output_node(output_n, partition, removed_nodes):
                transparent_output_nodes[node] = False
                return False
        transparent_output_nodes[node] = True
        return True
    transparent_output_nodes[node] = False
    return False