from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
@primitive
def tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim):
    if A_ndim == 0:
        return G * A
    G_axes = onp.arange(onp.ndim(G))
    if type(axes) is int:
        axes = max(axes, 0)
        A_axes = onp.arange(A_ndim)
        return onp.tensordot(A, G, [A_axes[:A_ndim - axes], G_axes[:A_ndim - axes]])
    else:
        axes0 = [axes[0]] if type(axes[0]) is int else axes[0]
        axes1 = [axes[1]] if type(axes[1]) is int else axes[1]
        axes = [axes0, axes1]
        A_axes = onp.arange(A_ndim)
        B_axes = onp.arange(B_ndim)
        summed_axes = [onp.asarray(axes[0], dtype='int64') % A_ndim, onp.asarray(axes[1], dtype='int64') % B_ndim]
        other_axes = [onp.delete(A_axes, summed_axes[0]), onp.delete(B_axes, summed_axes[1])]
        out = onp.tensordot(A, G, [other_axes[0], G_axes[:len(other_axes[0])]])
        perm = onp.argsort(onp.concatenate((summed_axes[1][onp.argsort(summed_axes[0])], other_axes[1])))
        return onp.transpose(out, perm)