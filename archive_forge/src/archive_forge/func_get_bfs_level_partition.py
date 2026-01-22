import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def get_bfs_level_partition(partitions: List[Partition]) -> None:
    """Given a list of partitions,
    mark the bfs level for each partition
    """
    current_level: Set[Partition] = set()
    visited: Set[Partition] = set()
    for partition in partitions:
        if len(partition.parents) == 0:
            current_level.add(partition)
    next_level: Set[Partition] = set()
    level = 0
    while current_level:
        partition = current_level.pop()
        partition.bfs_level = level
        visited.add(partition)
        children = partition.children
        for child in children:
            if child not in next_level:
                next_level.add(child)
        if not current_level:
            current_level = next_level.copy()
            next_level = set()
            level += 1
    return