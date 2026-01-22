import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def mm_args(mat1, mat2, *others, layout=None, out_dtype=None, use_4x2_dim=False):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout
        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        layout = FixedLayout(mat1.get_device(), out_dtype, [*b, m, n])
    else:
        assert out_dtype is None, 'out_dtype is ignored if layout is specified.'
    from ..lowering import expand
    others = [realize_inputs(expand(x, layout.size)) for x in others]
    return [m, n, k, layout, mat1, mat2, *others]