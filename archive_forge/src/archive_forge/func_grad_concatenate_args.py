from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_concatenate_args(argnum, ans, axis_args, kwargs):
    axis, args = (axis_args[0], axis_args[1:])
    sizes = [anp.shape(a)[axis] for a in args[:argnum]]
    start = sum(sizes[:-1])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + sizes[-1])
    return lambda g: g[tuple(idxs)]