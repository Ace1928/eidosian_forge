from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def fft_grad(get_args, fft_fun, ans, x, *args, **kwargs):
    axes, s, norm = get_args(x, *args, **kwargs)
    check_no_repeated_axes(axes)
    vs = vspace(x)
    return lambda g: match_complex(x, truncate_pad(fft_fun(g, *args, **kwargs), vs.shape))