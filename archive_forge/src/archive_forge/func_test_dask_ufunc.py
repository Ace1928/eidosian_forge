import array
import subprocess
import sys
import numpy as np
import pytest
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_dask_ufunc():
    olduse = pd.get_option('compute.use_numexpr')
    try:
        da = pytest.importorskip('dask.array')
        dd = pytest.importorskip('dask.dataframe')
        s = Series([1.5, 2.3, 3.7, 4.0])
        ds = dd.from_pandas(s, npartitions=2)
        result = da.fix(ds).compute()
        expected = np.fix(s)
        tm.assert_series_equal(result, expected)
    finally:
        pd.set_option('compute.use_numexpr', olduse)