from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
@primitive
def dot_adjoint_0(B, G, A_meta, B_meta):
    _, A_ndim, A_dtype, _ = A_meta
    _, B_ndim, _, _ = B_meta
    if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
        contract_num = max(0, B_ndim - (A_ndim != 0))
        out = onp.tensordot(G, B, contract_num)
    else:
        out = onp.tensordot(G, onp.swapaxes(B, -1, -2), B_ndim - 1)
    return onp.asarray(out, dtype=A_dtype)