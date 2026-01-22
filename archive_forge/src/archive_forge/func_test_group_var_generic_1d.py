import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_group_var_generic_1d(self):
    prng = np.random.default_rng(2)
    out = (np.nan * np.ones((5, 1))).astype(self.dtype)
    counts = np.zeros(5, dtype='int64')
    values = 10 * prng.random((15, 1)).astype(self.dtype)
    labels = np.tile(np.arange(5), (3,)).astype('intp')
    expected_out = (np.squeeze(values).reshape((5, 3), order='F').std(axis=1, ddof=1) ** 2)[:, np.newaxis]
    expected_counts = counts + 3
    self.algo(out, counts, values, labels)
    assert np.allclose(out, expected_out, self.rtol)
    tm.assert_numpy_array_equal(counts, expected_counts)