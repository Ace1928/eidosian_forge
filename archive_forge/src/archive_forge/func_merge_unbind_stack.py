import itertools
import logging
import operator
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import TypeAlias
import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
from .pre_grad import (
@register_graph_pattern(CallFunction([torch.stack, torch.cat], getitem_unbind, Ignored(), _users=MULTIPLE), pass_dict=unbind_stack_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction([torch.stack, torch.cat], getitem_unbind, dim=Ignored(), _users=MULTIPLE), pass_dict=unbind_stack_pass, extra_check=config_flag('split_cat_fx_passes'))
@register_graph_pattern(CallFunction([torch.stack, torch.cat], tensors=getitem_unbind, dim=Ignored(), _users=MULTIPLE), pass_dict=unbind_stack_pass, extra_check=config_flag('split_cat_fx_passes'))
def merge_unbind_stack(match: Match, unbind_input: torch.fx.Node, dim: int):
    unbind_node = next((node for node in match.nodes if node.target == torch.unbind))
    UnbindCatRemover().remove_unbind(match.graph, unbind_node)