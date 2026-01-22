from __future__ import annotations
import inspect
import warnings
from collections.abc import Sequence
import numpy as np
from dask.array.core import asarray
from dask.array.core import concatenate as _concatenate
from dask.array.creation import arange as _arange
from dask.array.numpy_compat import NUMPY_GE_200
from dask.utils import derived_from, skip_doctest
def fft_wrap(fft_func, kind=None, dtype=None):
    """Wrap 1D, 2D, and ND real and complex FFT functions

    Takes a function that behaves like ``numpy.fft`` functions and
    a specified kind to match it to that are named after the functions
    in the ``numpy.fft`` API.

    Supported kinds include:

        * fft
        * fft2
        * fftn
        * ifft
        * ifft2
        * ifftn
        * rfft
        * rfft2
        * rfftn
        * irfft
        * irfft2
        * irfftn
        * hfft
        * ihfft

    Examples
    --------
    >>> import dask.array.fft as dff
    >>> parallel_fft = dff.fft_wrap(np.fft.fft)
    >>> parallel_ifft = dff.fft_wrap(np.fft.ifft)
    """
    if scipy is not None:
        if fft_func is scipy.fftpack.rfft:
            raise ValueError("SciPy's `rfft` doesn't match the NumPy API.")
        elif fft_func is scipy.fftpack.irfft:
            raise ValueError("SciPy's `irfft` doesn't match the NumPy API.")
    if kind is None:
        kind = fft_func.__name__
    try:
        out_chunk_fn = _out_chunk_fns[kind.rstrip('2n')]
    except KeyError:
        raise ValueError('Given unknown `kind` %s.' % kind)

    def func(a, s=None, axes=None):
        a = asarray(a)
        if axes is None:
            if kind.endswith('2'):
                axes = (-2, -1)
            elif kind.endswith('n'):
                if s is None:
                    axes = tuple(range(a.ndim))
                else:
                    if NUMPY_GE_200:
                        warnings.warn('DeprecationWarning: `axes` should not be `None` if `s` is not `None` (Deprecated in NumPy 2.0)', DeprecationWarning)
                    axes = tuple(range(len(s)))
            else:
                axes = (-1,)
        elif len(set(axes)) < len(axes):
            raise ValueError('Duplicate axes not allowed.')
        _dtype = dtype
        if _dtype is None:
            sample = np.ones(a.ndim * (8,), dtype=a.dtype)
            try:
                _dtype = fft_func(sample, axes=axes).dtype
            except TypeError:
                _dtype = fft_func(sample).dtype
        for each_axis in axes:
            if len(a.chunks[each_axis]) != 1:
                raise ValueError(chunk_error % (each_axis, a.chunks[each_axis]))
        chunks = out_chunk_fn(a, s, axes)
        args = (s, axes)
        if kind.endswith('fft'):
            axis = None if axes is None else axes[0]
            n = None if s is None else s[0]
            args = (n, axis)
        return a.map_blocks(fft_func, *args, dtype=_dtype, chunks=chunks)
    if kind.endswith('fft'):
        _func = func

        def func(a, n=None, axis=None):
            s = None
            if n is not None:
                s = (n,)
            axes = None
            if axis is not None:
                axes = (axis,)
            return _func(a, s, axes)
    func_mod = inspect.getmodule(fft_func)
    func_name = fft_func.__name__
    func_fullname = func_mod.__name__ + '.' + func_name
    if fft_func.__doc__ is not None:
        func.__doc__ = fft_preamble % (2 * (func_fullname,))
        func.__doc__ += fft_func.__doc__
        func.__doc__ = skip_doctest(func.__doc__)
    func.__name__ = func_name
    return func