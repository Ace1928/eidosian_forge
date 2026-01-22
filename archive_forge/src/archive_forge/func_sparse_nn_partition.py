import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
def sparse_nn_partition(self, available_mem_bytes: int) -> None:
    """This method partition a sparse nn module.
        It is size based partition but different from size_based_partition,
        it only works when all the devices have same memory size (available_mem_bytes).
        In the future, devices with different mem sizes will be supported like size_based_partition.
        It first traverse all the nodes and do the partitions based on the same memory size.
        If the current partition has no enough memory left for a new op node
        (call_module, call_method, call_function), a new partition is created.
        When crossing the boundary between non-embedding nodes and embedding nodes,
        a new partition is created regardlessly.
        For example, if the current node is a non-embedding node but the next node is an
        embedding node, a new partition is created for the next node.
        After the partition, the partitions are combined as much as possible.
        The rule is that a non-embedding partition only
        combines with another non-embedding one.
        So as the embedding partitions.
        """

    def combine_partitions_based_on_size(partitions: List[Partition], available_mem_bytes: int) -> None:
        """Combining small partitions together to keep as less partitions as possible.
            Here is an example of the algorithm to do this:
            Assume some partitions, we first sort them based on partition used memory size.
            [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]
            The available memory is 10.
            step 1: self.find_partition_to_combine_based_on_size()
            First, mark bfs level for each partition
            Second, look the smallest partition, partition_4: 10 - 1 = 9
            It means any partition has a used memory equal or less than 9 could combine this partition
            We go from the largest and selection partition_0.
            Check the bfs level for two partitions, if the level difference is less than 2,
            it can be combined.
            step 2: repeat step 1 until no partitions can be combined
            """
        find_combination = True
        while find_combination:
            sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
            get_bfs_level_partition(self.partitions)
            find_combination, partitions = find_partition_to_combine_based_on_size(sorted_partitions, available_mem_bytes, partitions)
        return

    def calculate_mem_bytes_needed(p1, p2):
        """Given two partitions, calculate how many mem bytes
            are needed if two partitions are combined
            """
        nodes = p1.nodes.union(p2.nodes)
        mem_bytes_needed = 0
        for node in nodes:
            mem_bytes_needed += get_extra_size_of(node, nodes)
        return mem_bytes_needed

    def find_partition_to_combine_based_on_size(sorted_partitions: List[Partition], available_mem_bytes: int, partitions: List[Partition]) -> Tuple[bool, List[Partition]]:
        """step 1 in combine_partition_based_on_size()"""
        find_combination = False
        smallest_partition = sorted_partitions.pop(0)
        for p in sorted_partitions[::-1]:
            if abs(smallest_partition.bfs_level - p.bfs_level) <= 1:
                mem_bytes_needed = calculate_mem_bytes_needed(p, smallest_partition)
                if mem_bytes_needed <= available_mem_bytes:
                    combine_two_partitions(p, smallest_partition, self.partitions)
                    partitions.remove(smallest_partition)
                    partitions.remove(p)
                    partitions.append(self.partitions[-1])
                    find_combination = True
                    break
        return (find_combination, partitions)

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

    def is_embedding_node(node: Node) -> bool:
        """Check if a node is an embedding node"""
        if node.op == 'call_module':
            submodule = self.graph_module
            for atom in str(node.target).split('.'):
                if not hasattr(submodule, atom):
                    raise RuntimeError(f'Module {submodule} has no attribute {atom}')
                submodule = getattr(submodule, atom)
                if 'Embedding' in str(submodule):
                    return True
        return False
    embedding_partitions: List[Partition] = []
    non_embedding_partitions: List[Partition] = []
    in_embedding_region: bool = False
    partition = self.create_partition()
    for node in self.graph_module.graph.nodes:
        if node.op in {'call_module', 'call_method', 'call_function'}:
            if is_embedding_node(node) != in_embedding_region:
                if partition.used_mem_bytes != 0:
                    partition = reset_partition_in_sparse_nn(partition)
                in_embedding_region = not in_embedding_region
            total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
            if total_size_of_input_nodes + partition.used_mem_bytes > available_mem_bytes:
                partition = reset_partition_in_sparse_nn(partition)
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if total_size_of_input_nodes > available_mem_bytes:
                    raise RuntimeError(node.target + 'is too large to fit into a device')
            partition.add_node(node)
    reset_partition_in_sparse_nn(partition, new_partition=False)
    set_parents_and_children(self.partitions)
    combine_partitions_based_on_size(non_embedding_partitions, available_mem_bytes)
    combine_partitions_based_on_size(embedding_partitions, available_mem_bytes)
    total_size_of_non_embedding_partitions = 0
    for partition in non_embedding_partitions:
        total_size_of_non_embedding_partitions += partition.used_mem_bytes
    if len(embedding_partitions) > len(self.devices):
        msg = 'Need ' + str(len(embedding_partitions)) + ' devices, but only ' + str(len(self.devices)) + ' provided'
        raise RuntimeError(msg)
    occupied_devices = []
    for i, partition in enumerate(embedding_partitions):
        if total_size_of_non_embedding_partitions + partition.used_mem_bytes > available_mem_bytes:
            raise RuntimeError('partition_' + str(partition.partition_id) + '(embedding partition) and non embedding partitions can not fit into one device')
        else:
            partition.logical_device_ids = [self.devices[i].logical_id]
            occupied_devices.append(self.devices[i].logical_id)
    for partition in non_embedding_partitions:
        partition.logical_device_ids = occupied_devices
    self.node_to_partition = get_node_to_partition_mapping(self.partitions)
    return