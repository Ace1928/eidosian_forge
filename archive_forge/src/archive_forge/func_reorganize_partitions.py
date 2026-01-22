import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def reorganize_partitions(partitions: List[Partition]) -> None:
    """Given a list of partitions, reorganize partition id,
    its parents and its children for each partition
    """
    for i, partition in enumerate(partitions):
        partition.partition_id = i
    set_parents_and_children(partitions)
    return