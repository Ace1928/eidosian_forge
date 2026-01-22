from typing import Dict, List, Set, Iterable, Sequence, Optional, Deque
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupportBase
import logging
import itertools
from copy import copy
from collections import deque
def remove_bookend_non_compute_ops(self, partitions: List[Partition]):
    non_compute_ops = set(self.non_compute_ops)

    def is_non_compute_node(node: Node):
        return node.op == 'call_function' and _get_qualified_name(node.target) in non_compute_ops
    transparent_input_nodes: Dict[Node, bool] = {}
    transparent_output_nodes: Dict[Node, bool] = {}

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
    for partition in partitions:
        remove_node: Set[Node] = set()
        for node in partition.nodes:
            if is_non_compute_node(node) and (is_transparent_input_node(node, partition.nodes, remove_node) or is_transparent_output_node(node, partition.nodes, remove_node)):
                remove_node.add(node)
        if len(remove_node) != 0:
            partition.nodes = partition.nodes - remove_node