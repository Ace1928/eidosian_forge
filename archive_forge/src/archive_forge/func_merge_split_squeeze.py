import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(RepeatedExpr(CallFunction(torch.squeeze, GetItem(TorchSplit(KeywordArg('split_input'), KeywordArg('split_sizes')), Ignored()), KeywordArg('dim'), _users=MULTIPLE)), pass_dict=split_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(RepeatedExpr(CallFunction(torch.squeeze, GetItem(TorchSplit(KeywordArg('split_input'), KeywordArg('split_sizes')), Ignored()), dim=KeywordArg('dim'), _users=MULTIPLE)), pass_dict=split_cat_pass, extra_check=config_flag('split_cat_fx_passes'))
def merge_split_squeeze(match: Match, split_input: torch.fx.Node, split_sizes: List[int], dim: int):
    graph = match.graph
    split = next((node for node in match.nodes if node.target == torch.split))
    if not all((s == 1 for s in split_sizes)):
        return
    if isinstance(dim, Sequence):
        return
    next_users = find_next_users(split)
    if not all((node.target == torch.squeeze for node in next_users)):
        return
    with graph.inserting_before(match.output_node()):
        unbind = graph.call_function(torch.unbind, args=(split_input,), kwargs={'dim': dim})
        for item_index, getitem_node in sorted([(getitem_node.args[1], getitem_node) for getitem_node in split.users.keys()]):
            squeeze = next(iter(getitem_node.users.keys()))
            new_get_item = graph.call_function(operator.getitem, args=(unbind, item_index))
            squeeze.replace_all_uses_with(new_get_item)
            new_get_item.meta.update(squeeze.meta)
            graph.erase_node(squeeze)
            graph.erase_node(getitem_node)
    graph.erase_node(split)
    counters['inductor']['split_squeeze_replaced'] += 1