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
def test_concat_compat() -> None:
    ds1 = Dataset({'has_x_y': (('y', 'x'), [[1, 2]]), 'has_x': ('x', [1, 2]), 'no_x_y': ('z', [1, 2])}, coords={'x': [0, 1], 'y': [0], 'z': [-1, -2]})
    ds2 = Dataset({'has_x_y': (('y', 'x'), [[3, 4]]), 'has_x': ('x', [1, 2]), 'no_x_y': (('q', 'z'), [[1, 2]])}, coords={'x': [0, 1], 'y': [1], 'z': [-1, -2], 'q': [0]})
    result = concat([ds1, ds2], dim='y', data_vars='minimal', compat='broadcast_equals')
    assert_equal(ds2.no_x_y, result.no_x_y.transpose())
    for var in ['has_x', 'no_x_y']:
        assert 'y' not in result[var].dims and 'y' not in result[var].coords
    with pytest.raises(ValueError, match="'q' not present in all datasets"):
        concat([ds1, ds2], dim='q')
    with pytest.raises(ValueError, match="'q' not present in all datasets"):
        concat([ds2, ds1], dim='q')