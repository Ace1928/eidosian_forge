from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_chooser(ans, x, axis=None, keepdims=None):
    shape, dtype = (anp.shape(x), anp.result_type(x))

    def vjp(g):
        """Builds gradient of functions that choose a single item, such as min or max."""
        g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        argmax_locations = x == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0]
        return g_repeated * argmax_locations / onp.sum(argmax_locations, axis=axis, keepdims=True)
    return vjp