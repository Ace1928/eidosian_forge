import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def partial_lower(gm: torch.fx.GraphModule, node_predicate: Callable[[torch.fx.Node], bool]=lambda x: True, subgraph_predicate: Callable[[List[torch.fx.Node]], bool]=lambda x: True, dumper: Callable[[str], str]=lambda x: 'subgraph') -> torch.fx.GraphModule:
    """
    Lower Inductor compatible portions of the graph module to Inductor.

    Args:
        node_predicate: user predicate for determining whether to consider a node for
            lowering.
        subgraph_predicate: user predicate for determining whether to consider a list of
            candidate nodes for lowering.
        dumper: a callback for dumping subgraphs for human digestion. For exmaple, it
            can be a function that writes to disk/blob storage and returns the
            path/handle. The returned path/handle for each subgraph will be made
            available in the subgraph call node in the parent graph, as well as the
            label of the profiler block for the subgraph.
    """
    nodes_per_subgraph: List[List[torch.fx.Node]] = [[]]
    ptr = next(iter(gm.graph.nodes))

    def _node_predicate(node: torch.fx.Node) -> Tuple[bool, str]:
        should_lower, reason = _is_inductor_compatible(node)
        if not should_lower:
            return (should_lower, reason)
        if not node_predicate(node):
            return (False, 'user predicate')
        return (True, '')
    while ptr.op != 'output':
        if ptr.op == 'placeholder':
            ptr = ptr.next
            continue
        should_lower, reason = _node_predicate(ptr)
        if should_lower:
            nodes_per_subgraph[-1].append(ptr)
        else:
            if len(nodes_per_subgraph[-1]) > 0:
                logger.warning('partial_lower: graph break at %s. Reason: %s', str(ptr), reason)
            nodes_per_subgraph.append([])
        ptr = ptr.next
    nodes_per_subgraph = [nodes for nodes in nodes_per_subgraph if subgraph_predicate(nodes) and _subgraph_predicate(nodes)]
    for idx, subgraph_nodes in enumerate(nodes_per_subgraph):
        subgraph_name = f'subgraph_{idx}'
        _lower_subgraph_nodes(gm, subgraph_name, subgraph_nodes, dumper)
    gm.graph.lint()
    gm.recompile()
    return gm