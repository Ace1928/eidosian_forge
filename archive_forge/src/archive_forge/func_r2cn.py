import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
def r2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    """Return multidimensional discrete Fourier transform of real input"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet supported by scipy.fft functions')
    tmp = _asfarray(x)
    if not np.isrealobj(tmp):
        raise TypeError('x must be a real sequence')
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    tmp, _ = _fix_shape(tmp, shape, axes)
    norm = _normalization(norm, forward)
    workers = _workers(workers)
    if len(axes) == 0:
        raise ValueError('at least 1 axis must be transformed')
    return pfft.r2c(tmp, axes, forward, norm, None, workers)