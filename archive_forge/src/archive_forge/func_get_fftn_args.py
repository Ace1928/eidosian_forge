from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
def get_fftn_args(a, s=None, axes=None, norm=None, *args, **kwargs):
    if axes is None:
        axes = list(range(a.ndim))
    return (axes, s, norm)