from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if shape == ():
        return (g, 1)
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = onp.array(shape)
    new_shape[axis] = 1
    num_reps = onp.prod(onp.array(shape)[axis])
    return (anp.reshape(g, new_shape) + onp.zeros(shape, dtype=dtype), num_reps)