import collections
import logging
import operator
from typing import Any, DefaultDict, Deque, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch._dynamo.utils import counters
from torch._utils_internal import print_graph
from .. import config
from ..pattern_matcher import (
def is_linear_node_can_be_fused(node: torch.fx.Node):
    input = get_arg_value(node, 0, 'input')
    weight = get_arg_value(node, 1, 'weight')
    return is_node_meta_valid(node) and is_node_meta_valid(input) and is_node_meta_valid(weight) and (len(input.meta['example_value'].shape) == 2) and (len(weight.meta['example_value'].shape) == 2)