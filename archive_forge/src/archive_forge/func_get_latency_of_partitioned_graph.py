from enum import Enum
from typing import NamedTuple, Dict, List, Set
from torch.fx.node import Node, map_arg
def get_latency_of_partitioned_graph(partitions: List[Partition], partition_to_latency_mapping: Dict[Partition, PartitionLatency], transfer_rate_bytes_per_sec: float):
    """Given all partitions in a graph, find the critical path among all partitions
    and return its latency as the latency of the whole graph
    """

    def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
        """This function helps to recursively get the latency of a path of partitions"""
        latency_so_far_sec += partition_to_latency_mapping[partition].overall_latency_sec
        children = partition.children
        if partition.children:
            max_latency_sec = 0.0
            for child in partition.children:
                comm_latency_sec = get_comm_latency_between(partition, child, transfer_rate_bytes_per_sec)
                new_latency_sec = dfs_helper(child, latency_so_far_sec + comm_latency_sec)
                if new_latency_sec > max_latency_sec:
                    max_latency_sec = new_latency_sec
            return max_latency_sec
        return latency_so_far_sec

    def get_top_partitions(partitions: List[Partition]) -> List[Partition]:
        """This function is to return all the partitions without parents
        as the starting points of all the paths
        """
        top_partitions = []
        for partition in partitions:
            if len(partition.parents) == 0:
                top_partitions.append(partition)
        return top_partitions
    top_partitions = get_top_partitions(partitions)
    critical_path_latency_sec = 0.0
    for partition in top_partitions:
        latency_sec = dfs_helper(partition, 0.0)
        if latency_sec > critical_path_latency_sec:
            critical_path_latency_sec = latency_sec
    return critical_path_latency_sec