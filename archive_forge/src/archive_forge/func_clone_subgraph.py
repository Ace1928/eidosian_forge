import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def clone_subgraph(graph: fx.Graph, subgraph: List[fx.Node], target: fx.Node) -> List[fx.Node]:
    """Clone the given subgraph and insert it before ``target``.

    This API currently does not support inserting after ``target``.
    """
    all_nodes = set(subgraph)
    mapping: Dict[fx.Node, fx.Node] = dict()
    cloned_subgraph = []
    with graph.inserting_before(target):
        for node in subgraph:
            cloned_node = graph.call_function(node.target, node.args, node.kwargs, node.type)
            original_input = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            cloned_input, spec = tree_flatten((cloned_node.args, cloned_node.kwargs))
            mapped_cloned_input = []
            for original_input_node, cloned_input_node in zip(original_input, cloned_input):
                if isinstance(original_input_node, fx.Node) and original_input_node in all_nodes:
                    assert original_input_node in mapping
                    mapped_cloned_input.append(mapping[original_input_node])
                else:
                    mapped_cloned_input.append(cloned_input_node)
            cloned_node.args, cloned_node.kwargs = tree_unflatten(mapped_cloned_input, spec)
            mapping[node] = cloned_node
            cloned_subgraph.append(cloned_node)
    return cloned_subgraph