import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
def remove_split_with_size_one(graph: torch.fx.Graph, node: torch.fx.Node, input: torch.fx.Node):
    next_users = find_next_users(node)
    user = next(iter(node.users.keys()))
    for next_user in next_users:
        next_user.replace_input_with(user, input)
    graph.erase_node(user)
    graph.erase_node(node)
    counters['inductor']['remove_split_with_size_one'] += 1