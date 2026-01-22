from __future__ import annotations
import pickle
import re
import sys
import warnings
from collections.abc import Hashable
from copy import copy, deepcopy
from io import StringIO
from textwrap import dedent
from typing import Any, Literal
import numpy as np
import pandas as pd
import pytest
from pandas.core.indexes.datetimes import DatetimeIndex
import xarray as xr
from xarray import (
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import dtypes, indexing, utils
from xarray.core.common import duck_array_ops, full_like
from xarray.core.coordinates import Coordinates, DatasetCoordinates
from xarray.core.indexes import Index, PandasIndex
from xarray.core.utils import is_scalar
from xarray.namedarray.pycompat import array_type, integer_types
from xarray.testing import _assert_internal_invariants
from xarray.tests import (
def test_dataset_transpose(self) -> None:
    ds = Dataset({'a': (('x', 'y'), np.random.randn(3, 4)), 'b': (('y', 'x'), np.random.randn(4, 3))}, coords={'x': range(3), 'y': range(4), 'xy': (('x', 'y'), np.random.randn(3, 4))})
    actual = ds.transpose()
    expected = Dataset({'a': (('y', 'x'), ds.a.values.T), 'b': (('x', 'y'), ds.b.values.T)}, coords={'x': ds.x.values, 'y': ds.y.values, 'xy': (('y', 'x'), ds.xy.values.T)})
    assert_identical(expected, actual)
    actual = ds.transpose(...)
    expected = ds
    assert_identical(expected, actual)
    actual = ds.transpose('x', 'y')
    expected = ds.map(lambda x: x.transpose('x', 'y', transpose_coords=True))
    assert_identical(expected, actual)
    ds = create_test_data()
    actual = ds.transpose()
    for k in ds.variables:
        assert actual[k].dims[::-1] == ds[k].dims
    new_order = ('dim2', 'dim3', 'dim1', 'time')
    actual = ds.transpose(*new_order)
    for k in ds.variables:
        expected_dims = tuple((d for d in new_order if d in ds[k].dims))
        assert actual[k].dims == expected_dims
    new_order = ('dim2', 'dim3', 'dim1', 'time')
    actual = ds.transpose('dim2', 'dim3', ...)
    for k in ds.variables:
        expected_dims = tuple((d for d in new_order if d in ds[k].dims))
        assert actual[k].dims == expected_dims
    with pytest.raises(ValueError):
        ds.transpose(..., 'not_a_dim')
    actual = ds.transpose(..., 'not_a_dim', missing_dims='ignore')
    expected_ell = ds.transpose(...)
    assert_identical(expected_ell, actual)
    with pytest.warns(UserWarning):
        actual = ds.transpose(..., 'not_a_dim', missing_dims='warn')
        assert_identical(expected_ell, actual)
    assert 'T' not in dir(ds)