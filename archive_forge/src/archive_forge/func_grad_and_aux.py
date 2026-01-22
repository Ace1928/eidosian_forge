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
def grad_and_aux(fun, x):
    """Builds a function that returns the gradient of the first output and the
    (unmodified) second output of a function that returns two outputs."""
    vjp, (ans, aux) = _make_vjp(lambda x: atuple(fun(x)), x)
    return (vjp((vspace(ans).ones(), vspace(aux).zeros())), aux)