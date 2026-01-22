import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_constructor_from_sparse(self):
    zarr = SparseArray([0, 0, 1, 2, 3, 0, 4, 5, 0, 6], fill_value=0)
    res = SparseArray(zarr)
    assert res.fill_value == 0
    tm.assert_almost_equal(res.sp_values, zarr.sp_values)