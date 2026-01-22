import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def reset_partition_in_sparse_nn(partition, new_partition=True):
    """If crossing the boundary between non-embedding nodes and
            embedding nodes, create a new partition
            """
    if in_embedding_region:
        embedding_partitions.append(partition)
    else:
        non_embedding_partitions.append(partition)
    if new_partition:
        partition = self.create_partition()
        partition.left_mem_bytes = available_mem_bytes
        return partition
    return None