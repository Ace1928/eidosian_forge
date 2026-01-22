import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
@pytest.mark.parametrize('values, out', [([[np.inf], [np.inf], [np.inf]], [[np.inf], [np.inf]]), ([[np.inf], [np.inf], [-np.inf]], [[np.inf], [np.nan]]), ([[np.inf], [-np.inf], [np.inf]], [[np.inf], [np.nan]]), ([[np.inf], [-np.inf], [-np.inf]], [[np.inf], [-np.inf]])])
def test_cython_group_sum_Inf_at_begining_and_end(values, out):
    actual = np.array([[np.nan], [np.nan]], dtype='float64')
    counts = np.array([0, 0], dtype='int64')
    data = np.array(values, dtype='float64')
    labels = np.array([0, 1, 1], dtype=np.intp)
    group_sum(actual, counts, data, labels, None, is_datetimelike=False)
    expected = np.array(out, dtype='float64')
    tm.assert_numpy_array_equal(actual, expected)