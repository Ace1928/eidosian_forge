import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunctionVarArgs(torch.stack, users=MULTIPLE), pass_dict=normalization_pass, extra_check=config_flag('split_cat_fx_passes'))
def normalize_stack_default(match: Match, *args, **kwargs):
    node = match.nodes[0]
    graph = match.graph
    tensors = get_arg_value(node, 0, 'tensors')
    dim = get_arg_value(node, 1, 'dim') or 0
    if tensors is None or dim is None:
        log.info("couldn't find stack args")
        return
    assert isinstance(tensors, (list, tuple))
    for tensor in itertools.chain([node], tensors):
        if 'example_value' not in tensor.meta:
            log.warning('example value absent for node: %s', tensor)
            return
    ndim = node.meta['example_value'].dim()
    if dim < 0:
        dim += ndim
    with graph.inserting_after(node):
        new_node = graph.call_function(node.target, args=(tensors,), kwargs={'dim': dim})
    node.replace_all_uses_with(new_node)
    new_node.meta.update(node.meta)
    graph.erase_node(node)
    counters['inductor']['split_cat_norm'] += 1