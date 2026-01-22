from __future__ import absolute_import
from builtins import zip
import numpy.fft as ffto
from .numpy_wrapper import wrap_namespace
from .numpy_vjps import match_complex
from . import numpy_wrapper as anp
from autograd.extend import primitive, defvjp, vspace
@primitive
def truncate_pad(x, shape):
    slices = [slice(n) for n in shape]
    pads = tuple(zip(anp.zeros(len(shape), dtype=int), anp.maximum(0, anp.array(shape) - anp.array(x.shape))))
    return anp.pad(x, pads, 'constant')[tuple(slices)]