import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def calculate_extra_mem_bytes_needed_for(partition: Partition, partitions: List[Partition]):
    all_nodes: Set[Node] = set()
    for p in partitions:
        all_nodes = all_nodes.union(p.nodes)
    if len(all_nodes) == 0:
        return partition.used_mem_bytes
    all_nodes = all_nodes.union(partition.nodes)
    extra_size_needed = 0
    for node in partition.nodes:
        extra_size_needed += get_extra_size_of(node, all_nodes)
    return extra_size_needed