import functools
import torch
from torch._inductor.compile_fx import fake_tensor_prop
from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """
    from ..freezing import constant_fold
    lazy_init()
    binary_folding = counters['inductor']['binary_folding']
    fake_tensor_prop(gm, aot_example_inputs, True)
    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_convs(gm)
    for _ in range(4):
        constant_fold(gm)
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)
        if counters['inductor']['binary_folding'] == binary_folding:
            break
        binary_folding = counters['inductor']['binary_folding']
    torch._inductor.fx_passes.binary_folding.recover_original_precision_folded_convs(gm)
    constant_fold(gm)
    fake_tensor_prop(gm, aot_example_inputs, True)
    for pattern in pass_patterns:
        pattern.apply(gm.graph)
    if torch._C._has_mkldnn and config.cpp.weight_prepack and config.layout_optimization:
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes
        _eliminate_duplicate_packed_nodes(gm)
    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()