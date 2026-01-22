from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
def vector_dot_grad(*args, **kwargs):
    args, vector = (args[:-1], args[-1])
    return np.tensordot(fun_grad(*args, **kwargs), vector, np.ndim(vector))