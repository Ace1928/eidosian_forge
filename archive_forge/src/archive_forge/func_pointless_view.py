import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
@register_graph_pattern(CallFunction(torch.ops.aten.view.default, KeywordArg('arg'), KeywordArg('size')), pass_dict=patterns)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
    graph = match.graph
    node = match.output_node()
    arg_size = list(node.args[0].meta['val'].shape)
    if size == arg_size:
        node.replace_all_uses_with(node.args[0])
        match.erase_nodes(graph)