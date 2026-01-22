from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
import pytest
from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.coordinates import Coordinates
from xarray.core.indexes import PandasIndex
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
def test_concat_coords(self):
    data = Dataset({'foo': ('x', np.random.randn(10))})
    expected = data.assign_coords(c=('x', [0] * 5 + [1] * 5))
    objs = [data.isel(x=slice(5)).assign_coords(c=0), data.isel(x=slice(5, None)).assign_coords(c=1)]
    for coords in ['different', 'all', ['c']]:
        actual = concat(objs, dim='x', coords=coords)
        assert_identical(expected, actual)
    for coords in ['minimal', []]:
        with pytest.raises(merge.MergeError, match='conflicting values'):
            concat(objs, dim='x', coords=coords)