from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
from ..util import func
from .numpy_boxes import ArrayBox
def fwd_grad_sort(g, ans, x, axis=-1, kind='quicksort', order=None):
    sort_perm = anp.argsort(x, axis, kind, order)
    return g[sort_perm]