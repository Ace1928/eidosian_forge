from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def tensordot_vjp_0(ans, A, B, axes=2):
    A_ndim, B_ndim = (anp.ndim(A), anp.ndim(B))
    return lambda G: match_complex(A, tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim))