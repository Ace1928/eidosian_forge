from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
from ..util import func
from .numpy_boxes import ArrayBox
def fwd_grad_chooser(g, ans, x, axis=None, keepdims=False):
    if anp.isscalar(x):
        return g
    if not keepdims:
        if isinstance(axis, int):
            ans = anp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = anp.expand_dims(ans, ax)
    chosen_locations = x == ans
    return anp.sum(g * chosen_locations, axis=axis, keepdims=keepdims) / anp.sum(chosen_locations, axis=axis, keepdims=keepdims)