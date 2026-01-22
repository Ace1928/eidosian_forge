import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
def test_index_repr_bool_nan(self):
    arr = Index([True, False, np.nan], dtype=object)
    msg = 'Index.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        exp1 = arr.format()
    out1 = ['True', 'False', 'NaN']
    assert out1 == exp1
    exp2 = repr(arr)
    out2 = "Index([True, False, nan], dtype='object')"
    assert out2 == exp2