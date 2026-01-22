import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def find_single_partition(self, total_size_of_graph, logical_device_id: int=0) -> None:
    """Fit the whole fx module into one device"""
    partition_0 = self.create_partition()
    for node in self.graph_module.graph.nodes:
        if node.op == 'output':
            continue
        partition_0.nodes.add(node)
    partition_0.used_mem_bytes = total_size_of_graph
    partition_0.logical_device_ids = [logical_device_id]
    self.node_to_partition = get_node_to_partition_mapping(self.partitions)
    return