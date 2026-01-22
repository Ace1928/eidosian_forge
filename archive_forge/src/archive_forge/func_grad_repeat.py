from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_repeat(ans, x, repeats, axis=None):
    shape = anp.shape(x)

    def vjp(g):
        if axis is None:
            expanded = anp.reshape(g, (anp.prod(shape),) + (repeats,))
            return anp.reshape(anp.sum(expanded, axis=1, keepdims=False), shape)
        elif shape[axis] == 1:
            return anp.sum(g, axis=axis, keepdims=True)
        else:
            expanded = anp.reshape(g, shape[0:axis + 1] + (repeats,) + shape[axis + 1:])
            return anp.sum(expanded, axis=axis + 1, keepdims=False)
    return vjp