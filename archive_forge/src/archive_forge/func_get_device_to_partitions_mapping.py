import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def get_device_to_partitions_mapping(partitions: List[Partition], devices: List[Device]):
    """Given a list of partitions and a list of devices,
    map each partition into a device.
    """

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

    def find_device_for(partition: Partition):
        """Given a partition, find a logical device for the partition
        The algorithm is to put the partition on the device
        that has just enough mem left for that partition.
        device_to_left_mem_bytes is a dictionary between device and its left mem size
        sorted by its left mem size
        """
        for d in device_to_left_mem_bytes:
            extra_size_needed = calculate_extra_mem_bytes_needed_for(partition, device_to_partitions[d])
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False
    device_to_partitions, device_to_left_mem_bytes, no_device_partitions = get_device_partition_stats(partitions, devices)
    found_device = True
    for partition in no_device_partitions:
        device_to_left_mem_bytes = dict(sorted(device_to_left_mem_bytes.items(), key=lambda item: item[1]))
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device