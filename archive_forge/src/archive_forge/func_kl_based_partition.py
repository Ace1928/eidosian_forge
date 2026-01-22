import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def kl_based_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: Dict[Node, NodeLatency]) -> None:
    """This function is a cost aware partition based
        on Kernighan-Lin algorithm.
        First, the graph is partitioned using size_based_partition.
        Then, each node is swapped with any other node in a different
        partition, and at the same time, the cost is estimated after
        the swapping.
        For example, we have nodes n0, n1, n2, n3 and n4.
        Using size_based_partition, n0 and n1 are in Partition p0.
        n2, n3 and n4 in Partition p1. The current cost is estimated.
        We first tried using n0 to swap with n2 from the other partition.
        Then we see that swapping n0 and n2 shows a lower cost
        than the current cost and it is the minimum among other pairs like
        (n0, None)(This means moving n0 to Partition without swapping other nodes),
        (n0, n3) and (n0, n4). We swap n0 and n2 and set the new cost
        as the current cost.
        Then We repeat this process for all the other nodes until all swapping pairs
        are tried.
        """

    def swap_nodes(n0, n1, p0, p1):
        if n0 is not None:
            p0.remove_node(n0)
            p1.add_node(n0)
        if n1 is not None:
            p0.add_node(n1)
            p1.remove_node(n1)

    def try_swap_nodes(n0, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec):
        cost = float('inf')
        swap_nodes(n0, n1, p0, p1)
        reorganize_partitions(self.partitions)
        if not check_dependency(p0) and (not check_dependency(p1)):
            reset_partition_device(self.partitions)
            partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
            found_device = get_device_to_partitions_mapping(self.partitions, self.devices)
            if not found_device:
                cost = float('inf')
            else:
                cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
        swap_nodes(n1, n0, p0, p1)
        reorganize_partitions(self.partitions)
        reset_partition_device(self.partitions)
        get_device_to_partitions_mapping(self.partitions, self.devices)
        return cost

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
    self.size_based_partition()
    partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
    cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
    node_pair: List[Node] = []
    partition_pair: List[Partition] = []
    op_nodes = []
    for n in self.graph_module.graph.nodes:
        if n.op not in {'placeholder', 'get_attr', 'output'}:
            op_nodes.append(n)
    for node in op_nodes:
        p0_index = self.node_to_partition[node]
        p0 = self.partitions[p0_index]
        for p1_index, _ in enumerate(self.partitions):
            if p0_index != p1_index:
                p1 = self.partitions[p1_index]
                new_cost, new_node_pair = swap_node_to_partition(node, p0, p1, node_to_latency_mapping, transfer_rate_bytes_per_sec)
                if new_cost < cost:
                    cost = new_cost
                    node_pair = new_node_pair
                    partition_pair = [p0, p1]
        if len(node_pair) != 0:
            swap_nodes(node_pair[0], node_pair[1], partition_pair[0], partition_pair[1])
            reorganize_partitions(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
    reorganize_partitions(self.partitions)
    get_device_to_partitions_mapping(self.partitions, self.devices)
    return