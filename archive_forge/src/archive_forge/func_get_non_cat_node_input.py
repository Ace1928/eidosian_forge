import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def get_non_cat_node_input(self, split_node: torch.fx.Node, node: torch.fx.Node) -> List[_Range]:
    """
        Get input for a non cat node in the same format as `get_merged_user_inputs`
        """
    node_input = []
    split_users = set(split_node.users.keys())
    for node_arg in node.all_input_nodes:
        if node_arg in split_users:
            getitem_num = get_arg_value(node_arg, 1)
            node_input.append((getitem_num, getitem_num))
    return node_input