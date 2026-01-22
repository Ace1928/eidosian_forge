from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
from ..util import func
from .numpy_boxes import ArrayBox
def forward_grad_np_std(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = anp.size(g)
    elif isinstance(axis, int):
        num_reps = anp.shape(g)[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(anp.shape(g))[list(axis)])
    if num_reps <= 1:
        return anp.zeros_like(ans)
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) / ((num_reps - ddof) * ans)