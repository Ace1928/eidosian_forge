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
@register_graph_pattern(CallFunction(torch.ops.prims.convert_element_type.default, CallFunction(torch.ops.prims.convert_element_type.default, KeywordArg('arg'), KeywordArg('dtype1')), KeywordArg('dtype2')), pass_dict=patterns)
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """Remove chain of dtype conversions often created by AMP"""
    graph = match.graph
    node = match.output_node()
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if dtype1 in allowed and dtype2 in allowed:
        repl = graph.call_function(torch.ops.prims.convert_element_type.default, (arg, dtype2))
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes(graph)