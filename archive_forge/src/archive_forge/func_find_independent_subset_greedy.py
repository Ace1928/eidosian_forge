import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def find_independent_subset_greedy(node_list: List[torch.fx.Node], graph_search_options: Dict[str, Any]) -> Iterator[List[torch.fx.Node]]:
    """
    Return a list of subset from node_list, all nodes in each subset are independent with each other and can be fused together.
    The type of subset is list, so we can preserve node's order and benefit from split-cat elimination in later pass.
    """
    visited_node_set: Set[torch.fx.Node] = set()
    dep_set: Set[torch.fx.Node] = set()

    def find_dependent_nodes(src_node, cur_node):
        for input_node in cur_node.all_input_nodes:
            if input_node in node_list:
                dep_set.add(input_node)
            if input_node not in visited_node_set:
                visited_node_set.add(input_node)
                find_dependent_nodes(src_node, input_node)
    while len(node_list) > 0:
        subset: List[torch.fx.Node] = []
        subset_deps: Set[torch.fx.Node] = set()
        for node in node_list:
            if len(subset) >= graph_search_options['max_fuse_set_size']:
                break
            visited_node_set.clear()
            dep_set.clear()
            find_dependent_nodes(node, node)
            if not dep_set.intersection(subset) and node not in subset_deps:
                subset.append(node)
                subset_deps.update(dep_set)
        if len(subset) >= graph_search_options['min_fuse_set_size']:
            yield subset
        next_round_node_list = [node for node in node_list if node not in subset]
        node_list = next_round_node_list