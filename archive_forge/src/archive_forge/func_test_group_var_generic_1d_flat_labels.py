import numpy as np
import pytest
from pandas._libs import groupby as libgroupby
from pandas._libs.groupby import (
from pandas.core.dtypes.common import ensure_platform_int
from pandas import isna
import pandas._testing as tm
def test_group_var_generic_1d_flat_labels(self):
    prng = np.random.default_rng(2)
    out = (np.nan * np.ones((1, 1))).astype(self.dtype)
    counts = np.zeros(1, dtype='int64')
    values = 10 * prng.random((5, 1)).astype(self.dtype)
    labels = np.zeros(5, dtype='intp')
    expected_out = np.array([[values.std(ddof=1) ** 2]])
    expected_counts = counts + 5
    self.algo(out, counts, values, labels)
    assert np.allclose(out, expected_out, self.rtol)
    tm.assert_numpy_array_equal(counts, expected_counts)