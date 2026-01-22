import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging
import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility
from .operator_support import (
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
def find_reverse_deps(self, tag_id: Optional[int]=None) -> Dict[torch.fx.Node, NodeSet]:
    """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
    result: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue
        for user in node.users:
            if user.op not in CALLABLE_NODE_OPS:
                continue
            if tag_id is None or int(user.tag.split('_')[-1]) < tag_id:
                result[node].add(user)
    return result