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
def test_data_vars_properties(self) -> None:
    ds = Dataset()
    ds['foo'] = (('x',), [1.0])
    ds['bar'] = 2.0
    assert set(ds.data_vars) == {'foo', 'bar'}
    assert 'foo' in ds.data_vars
    assert 'x' not in ds.data_vars
    assert_identical(ds['foo'], ds.data_vars['foo'])
    expected = dedent('        Data variables:\n            foo      (x) float64 8B 1.0\n            bar      float64 8B 2.0')
    actual = repr(ds.data_vars)
    assert expected == actual
    assert ds.data_vars.dtypes == {'foo': np.dtype('float64'), 'bar': np.dtype('float64')}
    ds.coords['x'] = [1]
    assert len(ds.data_vars) == 2
    with pytest.raises(AssertionError, match='something is wrong with Dataset._coord_names'):
        ds._coord_names = {'w', 'x', 'y', 'z'}
        len(ds.data_vars)