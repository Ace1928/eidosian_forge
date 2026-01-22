from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.indexes import (
from xarray.core.variable import IndexVariable, Variable
from xarray.tests import assert_array_equal, assert_identical, requires_cftime
from xarray.tests.test_coding_times import _all_cftime_date_types
def test_reindex_like(self) -> None:
    index1 = PandasIndex([0, 1, 2], 'x')
    index2 = PandasIndex([1, 2, 3, 4], 'x')
    expected = {'x': [1, 2, -1, -1]}
    actual = index1.reindex_like(index2)
    assert actual.keys() == expected.keys()
    np.testing.assert_array_equal(actual['x'], expected['x'])
    index3 = PandasIndex([1, 1, 2], 'x')
    with pytest.raises(ValueError, match='.*index has duplicate values'):
        index3.reindex_like(index2)