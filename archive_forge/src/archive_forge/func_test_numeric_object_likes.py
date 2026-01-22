from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('case', [np.array([1, 2, 1, 5, 3, 2, 4, 1, 5, 6]), np.array([1.1, 2.2, 1.1, np.nan, 3.3, 2.2, 4.4, 1.1, np.nan, 6.6]), np.array([1 + 1j, 2 + 2j, 1 + 1j, 5 + 5j, 3 + 3j, 2 + 2j, 4 + 4j, 1 + 1j, 5 + 5j, 6 + 6j]), np.array(['a', 'b', 'a', 'e', 'c', 'b', 'd', 'a', 'e', 'f'], dtype=object), np.array([1, 2 ** 63, 1, 3 ** 5, 10, 2 ** 63, 39, 1, 3 ** 5, 7], dtype=np.uint64)])
def test_numeric_object_likes(self, case):
    exp_first = np.array([False, False, True, False, False, True, False, True, True, False])
    exp_last = np.array([True, True, True, True, False, False, False, False, False, False])
    exp_false = exp_first | exp_last
    res_first = algos.duplicated(case, keep='first')
    tm.assert_numpy_array_equal(res_first, exp_first)
    res_last = algos.duplicated(case, keep='last')
    tm.assert_numpy_array_equal(res_last, exp_last)
    res_false = algos.duplicated(case, keep=False)
    tm.assert_numpy_array_equal(res_false, exp_false)
    for idx in [Index(case), Index(case, dtype='category')]:
        res_first = idx.duplicated(keep='first')
        tm.assert_numpy_array_equal(res_first, exp_first)
        res_last = idx.duplicated(keep='last')
        tm.assert_numpy_array_equal(res_last, exp_last)
        res_false = idx.duplicated(keep=False)
        tm.assert_numpy_array_equal(res_false, exp_false)
    for s in [Series(case), Series(case, dtype='category')]:
        res_first = s.duplicated(keep='first')
        tm.assert_series_equal(res_first, Series(exp_first))
        res_last = s.duplicated(keep='last')
        tm.assert_series_equal(res_last, Series(exp_last))
        res_false = s.duplicated(keep=False)
        tm.assert_series_equal(res_false, Series(exp_false))