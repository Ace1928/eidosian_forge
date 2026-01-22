import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
@register_binary_folding_pattern(CallFunction(binary_op, _computation_call, KeywordArg('other')), extra_check=_is_foldable_pattern)
def folded_op(match, *args, **kwargs):
    counters['inductor']['binary_folding'] += 1
    other = kwargs.get('other')
    binary_node = match.output_node()
    computation_node = binary_node.args[0] if binary_node.args[0].target in _computation_ops else binary_node.args[1]
    graph = match.graph
    with graph.inserting_before(binary_node):
        assert computation_node.target == aten.convolution.default
        new_computation_node = _create_new_conv_node(graph, computation_node, binary_node, other)
        binary_node.replace_all_uses_with(new_computation_node)
        new_computation_node.meta.update(computation_node.meta)
        graph.erase_node(binary_node)
        graph.erase_node(computation_node)