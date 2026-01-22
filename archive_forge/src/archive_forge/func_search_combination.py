import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def search_combination(transfer_rate_bytes_per_sec, node_to_latency_mapping) -> bool:
    """Given transfer rate between partitions and each node's latency,
            find two partitions to combine so the cost of the partitions can
            be reduced.
            The algorithm is :
            1. Go through all the partition pairs and see
            if any pair of partitions can be combined.
            2. Calculate the cost after the combination.
            3. Select the minimum cost and combine its corresponding partition pair.
            """
    partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
    cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
    if len(self.partitions) == 1:
        return False
    partition_pair: List[int] = []
    for i in range(len(self.partitions) - 1):
        for j in range(i + 1, len(self.partitions)):
            new_cost = try_combining_partitions(i, j, self.partitions[:])
            if new_cost <= cost:
                partition_pair = [i, j]
                cost = new_cost
            reorganize_partitions(self.partitions)
    if len(partition_pair) != 0:
        p0 = self.partitions[partition_pair[0]]
        p1 = self.partitions[partition_pair[1]]
        combine_two_partitions(p0, p1, self.partitions)
    get_bfs_level_partition(self.partitions)
    reset_partition_device(self.partitions)
    get_device_to_partitions_mapping(self.partitions, self.devices)
    return len(partition_pair) != 0