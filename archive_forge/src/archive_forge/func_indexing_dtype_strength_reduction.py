import math
import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges
from .ir import LoopBody
from .utils import dominated_nodes
def indexing_dtype_strength_reduction(loop_body: LoopBody):
    """
    Performs Value Range Analysis on LoopBody's fx graph to reduce precision of
    intermediaries from int64 to int32
    """
    bv = loop_body.bounds()
    int64_dtype_nodes = [node for node in loop_body.get_nodes() if node.target == 'to_dtype' and node.args[2] == torch.int64 and (node not in bv.unbounded_vars)]
    if not int64_dtype_nodes:
        return
    bounds = bv.get_bounds()
    for node in int64_dtype_nodes:
        try_to_reduce_precision(node, bounds, loop_body.indirect_vars, loop_body.indexing_exprs, bv.replacement_vals)