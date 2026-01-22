import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_group_ohlc(dtype):
    obj = np.array(np.random.default_rng(2).standard_normal(20), dtype=dtype)
    bins = np.array([6, 12, 20])
    out = np.zeros((3, 4), dtype)
    counts = np.zeros(len(out), dtype=np.int64)
    labels = ensure_platform_int(np.repeat(np.arange(3), np.diff(np.r_[0, bins])))
    func = libgroupby.group_ohlc
    func(out, counts, obj[:, None], labels)

    def _ohlc(group):
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group[0], group.max(), group.min(), group[-1]]
    expected = np.array([_ohlc(obj[:6]), _ohlc(obj[6:12]), _ohlc(obj[12:])])
    tm.assert_almost_equal(out, expected)
    tm.assert_numpy_array_equal(counts, np.array([6, 6, 8], dtype=np.int64))
    obj[:6] = np.nan
    func(out, counts, obj[:, None], labels)
    expected[0] = np.nan
    tm.assert_almost_equal(out, expected)