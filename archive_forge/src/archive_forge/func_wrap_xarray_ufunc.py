import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
@conditional_dask
def wrap_xarray_ufunc(ufunc, *datasets, ufunc_kwargs=None, func_args=None, func_kwargs=None, dask_kwargs=None, **kwargs):
    """Wrap make_ufunc with xarray.apply_ufunc.

    Parameters
    ----------
    ufunc : callable
    *datasets : xarray.Dataset
    ufunc_kwargs : dict
        Keyword arguments passed to `make_ufunc`.
            - 'n_dims', int, by default 2
            - 'n_output', int, by default 1
            - 'n_input', int, by default len(datasets)
            - 'index', slice, by default Ellipsis
            - 'ravel', bool, by default True
    func_args : tuple
        Arguments passed to 'ufunc'.
    func_kwargs : dict
        Keyword arguments passed to 'ufunc'.
            - 'out_shape', int, by default None
    dask_kwargs : dict
        Dask related kwargs passed to :func:`xarray:xarray.apply_ufunc`.
        Use ``enable_dask`` method of :class:`arviz.Dask` to set default kwargs.
    **kwargs
        Passed to :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.Dataset
    """
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault('n_input', len(datasets))
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}
    if dask_kwargs is None:
        dask_kwargs = {}
    kwargs.setdefault('input_core_dims', tuple((('chain', 'draw') for _ in range(len(func_args) + len(datasets)))))
    ufunc_kwargs.setdefault('n_dims', len(kwargs['input_core_dims'][-1]))
    kwargs.setdefault('output_core_dims', tuple(([] for _ in range(ufunc_kwargs.get('n_output', 1)))))
    callable_ufunc = make_ufunc(ufunc, **ufunc_kwargs)
    return apply_ufunc(callable_ufunc, *datasets, *func_args, kwargs=func_kwargs, **dask_kwargs, **kwargs)