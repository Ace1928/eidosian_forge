import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('dtype, engine_type', [(np.int8, libindex.Int8Engine), (np.int16, libindex.Int16Engine), (np.int32, libindex.Int32Engine), (np.int64, libindex.Int64Engine)])
def test_engine_type(self, dtype, engine_type):
    if dtype != np.int64:
        num_uniques = {np.int8: 1, np.int16: 128, np.int32: 32768}[dtype]
        ci = CategoricalIndex(range(num_uniques))
    else:
        ci = CategoricalIndex(range(32768))
        arr = ci.values._ndarray.astype('int64')
        NDArrayBacked.__init__(ci._data, arr, ci.dtype)
    assert np.issubdtype(ci.codes.dtype, dtype)
    assert isinstance(ci._engine, engine_type)