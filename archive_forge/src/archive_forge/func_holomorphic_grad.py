from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
@unary_to_nary
def holomorphic_grad(fun, x):
    if not vspace(x).iscomplex:
        warnings.warn('Input to holomorphic_grad is not complex')
    return grad(lambda x: np.real(fun(x)))(x)