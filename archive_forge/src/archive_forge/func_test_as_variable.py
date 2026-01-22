from __future__ import annotations
import warnings
from abc import ABC
from copy import copy, deepcopy
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Generic
import numpy as np
import pandas as pd
import pytest
import pytz
from xarray import DataArray, Dataset, IndexVariable, Variable, set_options
from xarray.core import dtypes, duck_array_ops, indexing
from xarray.core.common import full_like, ones_like, zeros_like
from xarray.core.indexing import (
from xarray.core.types import T_DuckArray
from xarray.core.utils import NDArrayMixin
from xarray.core.variable import as_compatible_data, as_variable
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
from xarray.tests.test_namedarray import NamedArraySubclassobjects
def test_as_variable(self):
    data = np.arange(10)
    expected = Variable('x', data)
    expected_extra = Variable('x', data, attrs={'myattr': 'val'}, encoding={'scale_factor': 1})
    assert_identical(expected, as_variable(expected))
    ds = Dataset({'x': expected})
    var = as_variable(ds['x']).to_base_variable()
    assert_identical(expected, var)
    assert not isinstance(ds['x'], Variable)
    assert isinstance(as_variable(ds['x']), Variable)
    xarray_tuple = (expected_extra.dims, expected_extra.values, expected_extra.attrs, expected_extra.encoding)
    assert_identical(expected_extra, as_variable(xarray_tuple))
    with pytest.raises(TypeError, match='tuple of form'):
        as_variable(tuple(data))
    with pytest.raises(ValueError, match='tuple of form'):
        as_variable(('five', 'six', 'seven'))
    with pytest.raises(TypeError, match='without an explicit list of dimensions'):
        as_variable(data)
    with pytest.warns(FutureWarning, match='IndexVariable'):
        actual = as_variable(data, name='x')
    assert_identical(expected.to_index_variable(), actual)
    actual = as_variable(0)
    expected = Variable([], 0)
    assert_identical(expected, actual)
    data = np.arange(9).reshape((3, 3))
    expected = Variable(('x', 'y'), data)
    with pytest.raises(ValueError, match='without explicit dimension names'):
        as_variable(data, name='x')
    actual = as_variable(expected, name='x')
    assert_identical(expected, actual)
    dt = np.array([datetime(1999, 1, 1) + timedelta(days=x) for x in range(10)])
    with pytest.warns(FutureWarning, match='IndexVariable'):
        assert as_variable(dt, 'time').dtype.kind == 'M'
    td = np.array([timedelta(days=x) for x in range(10)])
    with pytest.warns(FutureWarning, match='IndexVariable'):
        assert as_variable(td, 'time').dtype.kind == 'm'
    with pytest.raises(TypeError):
        as_variable(('x', DataArray([])))