from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def set_dims(self, dims, shape=None):
    """Return a new variable with given set of dimensions.
        This method might be used to attach new dimension(s) to variable.

        When possible, this operation does not copy this variable's data.

        Parameters
        ----------
        dims : str or sequence of str or dict
            Dimensions to include on the new variable. If a dict, values are
            used to provide the sizes of new dimensions; otherwise, new
            dimensions are inserted with length 1.

        Returns
        -------
        Variable
        """
    if isinstance(dims, str):
        dims = [dims]
    if shape is None and is_dict_like(dims):
        shape = dims.values()
    missing_dims = set(self.dims) - set(dims)
    if missing_dims:
        raise ValueError(f'new dimensions {dims!r} must be a superset of existing dimensions {self.dims!r}')
    self_dims = set(self.dims)
    expanded_dims = tuple((d for d in dims if d not in self_dims)) + self.dims
    if self.dims == expanded_dims:
        expanded_data = self.data
    elif shape is not None:
        dims_map = dict(zip(dims, shape))
        tmp_shape = tuple((dims_map[d] for d in expanded_dims))
        expanded_data = duck_array_ops.broadcast_to(self.data, tmp_shape)
    else:
        indexer = (None,) * (len(expanded_dims) - self.ndim) + (...,)
        expanded_data = self.data[indexer]
    expanded_var = Variable(expanded_dims, expanded_data, self._attrs, self._encoding, fastpath=True)
    return expanded_var.transpose(*dims)