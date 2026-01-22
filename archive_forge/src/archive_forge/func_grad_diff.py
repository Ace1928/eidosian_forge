from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_diff(ans, a, n=1, axis=-1):
    nd = anp.ndim(a)
    ans_shape = anp.shape(ans)
    sl1 = [slice(None)] * nd
    sl1[axis] = slice(None, 1)
    sl2 = [slice(None)] * nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return anp.concatenate((-g[tuple(sl1)], -anp.diff(g, axis=axis), g[tuple(sl2)]), axis=axis)
        shape = list(ans_shape)
        shape[axis] = 1
        return anp.zeros(shape)

    def helper(g, n):
        if n == 0:
            return g
        return helper(undiff(g), n - 1)
    return lambda g: helper(g, n)