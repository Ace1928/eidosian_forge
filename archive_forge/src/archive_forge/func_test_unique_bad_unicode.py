import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.single_cpu
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='decoding fails')
def test_unique_bad_unicode(index_or_series):
    uval = '\ud83d'
    obj = index_or_series([uval] * 2)
    result = obj.unique()
    if isinstance(obj, pd.Index):
        expected = pd.Index(['\ud83d'], dtype=object)
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(['\ud83d'], dtype=object)
        tm.assert_numpy_array_equal(result, expected)