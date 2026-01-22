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
def pytree_to_dataset(data, *, attrs=None, library=None, coords=None, dims=None, default_dims=None, index_origin=None, skip_event_dims=None):
    """Convert a dictionary or pytree of numpy arrays to an xarray.Dataset.

    See https://jax.readthedocs.io/en/latest/pytrees.html for what a pytree is, but
    this inclues at least dictionaries and tuple types.

    Parameters
    ----------
    data : dict of {str : array_like or dict} or pytree
        Data to convert. Keys are variable names.
    attrs : dict, optional
        Json serializable metadata to attach to the dataset, in addition to defaults.
    library : module, optional
        Library used for performing inference. Will be attached to the attrs metadata.
    coords : dict of {str : ndarray}, optional
        Coordinates for the dataset
    dims : dict of {str : list of str}, optional
        Dimensions of each variable. The keys are variable names, values are lists of
        coordinates.
    default_dims : list of str, optional
        Passed to :py:func:`numpy_to_data_array`
    index_origin : int, optional
        Passed to :py:func:`numpy_to_data_array`
    skip_event_dims : bool, optional
        If True, cut extra dims whenever present to match the shape of the data.
        Necessary for PPLs which have the same name in both observed data and log
        likelihood groups, to account for their different shapes when observations are
        multivariate.

    Returns
    -------
    xarray.Dataset
        In case of nested pytrees, the variable name will be a tuple of individual names.

    Notes
    -----
    This function is available through two aliases: ``dict_to_dataset`` or ``pytree_to_dataset``.

    Examples
    --------
    Convert a dictionary with two 2D variables to a Dataset.

    .. ipython::

        In [1]: import arviz as az
           ...: import numpy as np
           ...: az.dict_to_dataset({'x': np.random.randn(4, 100), 'y': np.random.rand(4, 100)})

    Note that unlike the :class:`xarray.Dataset` constructor, ArviZ has added extra
    information to the generated Dataset such as default dimension names for sampled
    dimensions and some attributes.

    The function is also general enough to work on pytrees such as nested dictionaries:

    .. ipython::

        In [1]: az.pytree_to_dataset({'top': {'second': 1.}, 'top2': 1.})

    which has two variables (as many as leafs) named ``('top', 'second')`` and ``top2``.

    Dimensions and co-ordinates can be defined as usual:

    .. ipython::

        In [1]: datadict = {
           ...:     "top": {"a": np.random.randn(100), "b": np.random.randn(1, 100, 10)},
           ...:     "d": np.random.randn(100),
           ...: }
           ...: az.dict_to_dataset(
           ...:     datadict,
           ...:     coords={"c": np.arange(10)},
           ...:     dims={("top", "b"): ["c"]}
           ...: )

    """
    if dims is None:
        dims = {}
    try:
        data = {k[0] if len(k) == 1 else k: v for k, v in _flatten_with_path(data)}
    except TypeError:
        pass
    data_vars = {key: numpy_to_data_array(values, var_name=key, coords=coords, dims=dims.get(key), default_dims=default_dims, index_origin=index_origin, skip_event_dims=skip_event_dims) for key, values in data.items()}
    return xr.Dataset(data_vars=data_vars, attrs=make_attrs(attrs=attrs, library=library))