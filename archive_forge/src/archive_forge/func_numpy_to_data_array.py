import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import tree
import xarray as xr
from .. import __version__, utils
from ..rcparams import rcParams
def numpy_to_data_array(ary, *, var_name='data', coords=None, dims=None, default_dims=None, index_origin=None, skip_event_dims=None):
    """Convert a numpy array to an xarray.DataArray.

    By default, the first two dimensions will be (chain, draw), and any remaining
    dimensions will be "shape".
    * If the numpy array is 1d, this dimension is interpreted as draw
    * If the numpy array is 2d, it is interpreted as (chain, draw)
    * If the numpy array is 3 or more dimensions, the last dimensions are kept as shapes.

    To modify this behaviour, use ``default_dims``.

    Parameters
    ----------
    ary : np.ndarray
        A numpy array. If it has 2 or more dimensions, the first dimension should be
        independent chains from a simulation. Use `np.expand_dims(ary, 0)` to add a
        single dimension to the front if there is only 1 chain.
    var_name : str
        If there are no dims passed, this string is used to name dimensions
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : List(str)
        A list of coordinate names for the variable
    default_dims : list of str, optional
        Passed to :py:func:`generate_dims_coords`. Defaults to ``["chain", "draw"]``, and
        an empty list is accepted
    index_origin : int, optional
        Passed to :py:func:`generate_dims_coords`
    skip_event_dims : bool

    Returns
    -------
    xr.DataArray
        Will have the same data as passed, but with coordinates and dimensions
    """
    if default_dims is None:
        default_dims = ['chain', 'draw']
    if 'chain' in default_dims and 'draw' in default_dims:
        ary = utils.two_de(ary)
        n_chains, n_samples, *_ = ary.shape
        if n_chains > n_samples:
            warnings.warn('More chains ({n_chains}) than draws ({n_samples}). Passed array should have shape (chains, draws, *shape)'.format(n_chains=n_chains, n_samples=n_samples), UserWarning)
    else:
        ary = utils.one_de(ary)
    dims, coords = generate_dims_coords(ary.shape[len(default_dims):], var_name, dims=dims, coords=coords, default_dims=default_dims, index_origin=index_origin, skip_event_dims=skip_event_dims)
    if 'draw' not in dims and 'draw' in default_dims:
        dims = ['draw'] + dims
    if 'chain' not in dims and 'chain' in default_dims:
        dims = ['chain'] + dims
    index_origin = rcParams['data.index_origin']
    if 'chain' not in coords and 'chain' in default_dims:
        coords['chain'] = np.arange(index_origin, n_chains + index_origin)
    if 'draw' not in coords and 'draw' in default_dims:
        coords['draw'] = np.arange(index_origin, n_samples + index_origin)
    coords = {key: xr.IndexVariable((key,), data=np.asarray(coords[key])) for key in dims}
    return xr.DataArray(ary, coords=coords, dims=dims)