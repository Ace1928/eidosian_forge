import functools
import itertools
import torch
from ..._dynamo.utils import counters
from ..pattern_matcher import Arg, CallFunction, KeywordArg
from .freezing_patterns import register_binary_folding_pattern
def recover_original_precision_folded_convs(gm):
    """
    After binary folding conv weights and biases to a higher dtype, recover the original precision they were in.
    """
    graph = gm.graph
    convs = [node for node in graph.nodes if node.target is aten.convolution.default]
    for node in convs:
        orig_dtype = node.meta.get('_allow_conv_mixed_dtype_folding', None)
        if orig_dtype is None:
            continue
        with graph.inserting_before(node):
            for idx in [1, 2]:
                old_input = node.args[idx]
                if old_input is None:
                    continue
                new_input = graph.create_node('call_function', prims.convert_element_type.default, (old_input, orig_dtype))
                node.replace_input_with(old_input, new_input)