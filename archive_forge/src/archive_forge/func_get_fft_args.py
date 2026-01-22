from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def get_fft_args(a, d=None, axis=-1, norm=None, *args, **kwargs):
    axes = [axis]
    if d is not None:
        d = [d]
    return (axes, d, norm)