from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def pad_vjp(ans, array, pad_width, mode, **kwargs):
    assert mode == 'constant', 'Only constant mode padding is supported.'
    return lambda g: _unpad(g, pad_width)