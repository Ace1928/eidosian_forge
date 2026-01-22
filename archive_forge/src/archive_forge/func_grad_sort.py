from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_sort(ans, x, axis=-1, kind='quicksort', order=None):
    if len(x.shape) > 1:
        raise NotImplementedError('Gradient of sort not implemented for multi-dimensional arrays.')
    sort_perm = anp.argsort(x, axis, kind, order)
    return lambda g: unpermuter(g, sort_perm)