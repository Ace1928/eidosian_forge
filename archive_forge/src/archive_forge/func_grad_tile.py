from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_tile(ans, x, reps):
    reps = [reps] if anp.isscalar(reps) else reps
    x_shape = anp.shape(x)

    def vjp(g):
        for axis, rep in enumerate(reps):
            g = sum(anp.split(g, rep, axis))
        return anp.reshape(g, x_shape)
    return vjp