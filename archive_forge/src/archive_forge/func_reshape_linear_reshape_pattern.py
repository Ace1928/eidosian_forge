import functools
import operator
from functools import reduce
from typing import Any, Tuple
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from .. import ir
from ..lowering import lowerings as L
from ..pattern_matcher import (
from ..virtualized import ops
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
@register_freezing_graph_pattern(CallFunction(aten.reshape.default, CallFunction(mkldnn._linear_pointwise.default, CallFunction(aten.reshape.default, Arg(), KeywordArg('reshape_1'), _users=MULTIPLE), Arg(), Arg(), Arg(), Arg(), Arg()), KeywordArg('reshape_2')), pass_number=1)
def reshape_linear_reshape_pattern(match, *args, **kwargs):
    reshape_1 = kwargs.get('reshape_1')
    reshape_2 = kwargs.get('reshape_2')
    assert isinstance(reshape_1, list)
    assert isinstance(reshape_2, list)
    assert len(reshape_1) == 2
    dynamic_shapes = not all((isinstance(x, int) for x in [reshape_1[0]] + reshape_2[:-1]))
    graph = match.graph
    reshape_2_node = match.output_node()
    linear_input_node = reshape_2_node.args[0].args[0].args[0]
    if dynamic_shapes:
        return
    else:
        can_remove_reshape = linear_input_node.meta.get('val').shape[:-1] == torch.Size(reshape_2[:-1])
        can_remove_reshape = can_remove_reshape and reduce(lambda x, y: x * y, reshape_2[:-1]) == reshape_1[0]
    if can_remove_reshape:
        repl = graph.call_function(mkldnn._linear_pointwise.default, args)
        repl.meta.update(reshape_2_node.meta)
        reshape_2_node.replace_all_uses_with(repl)
        old_linear_node = reshape_2_node.args[0]
        reshape_1_node = old_linear_node.args[0]
        graph.erase_node(reshape_2_node)
        graph.erase_node(old_linear_node)
        if len(reshape_1_node.users) == 0:
            graph.erase_node(reshape_1_node)