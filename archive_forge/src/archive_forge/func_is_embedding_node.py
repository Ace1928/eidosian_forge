import operator
from collections import deque
from typing import Dict, List, Set, NamedTuple, Tuple, Deque
import torch
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.experimental.partitioner_utils import (
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from torch.fx.passes.split_module import split_module
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