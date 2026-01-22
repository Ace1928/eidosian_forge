from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type, Optional, Callable
import logging
import os
@compatibility(is_backward_compatible=False)
def get_source_partitions(graph: Graph, wanted_sources: List[Any], filter_fn: Optional[Callable[[Node], bool]]=None) -> Dict[Any, List[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.
    """
    modules: Dict[Type, Dict[str, List[Node]]] = {}
    for node in graph.nodes:
        if (source_fn_st := node.meta.get('source_fn_stack', None)) is None:
            continue
        source_fn = source_fn_st[-1]
        if source_fn[1] not in wanted_sources:
            continue
        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node)

    def make_partition(nodes: List[Node], module_type: Type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes:
                    input_nodes.add(arg)
            if node.op == 'get_attr':
                params.add(node)
            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)
        return SourcePartition(nodes, module_type, list(input_nodes), list(output_nodes), list(params))
    ret: Dict[Type[Any], List[SourcePartition]] = {}
    if filter_fn:
        filtered_modules = {}
        for tp, name_to_partition in modules.items():
            filtered_name_to_partition = {name: partition for name, partition in name_to_partition.items() if all(map(filter_fn, partition))}
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules
    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]
    return ret