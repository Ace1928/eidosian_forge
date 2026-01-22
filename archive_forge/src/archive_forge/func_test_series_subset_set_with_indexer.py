import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('indexer', [slice(0, 2), np.array([True, True, False]), np.array([0, 1])], ids=['slice', 'mask', 'array'])
def test_series_subset_set_with_indexer(backend, indexer_si, indexer, using_copy_on_write, warn_copy_on_write):
    _, _, Series = backend
    s = Series([1, 2, 3], index=['a', 'b', 'c'])
    s_orig = s.copy()
    subset = s[:]
    warn = None
    msg = 'Series.__setitem__ treating keys as positions is deprecated'
    if indexer_si is tm.setitem and isinstance(indexer, np.ndarray) and (indexer.dtype.kind == 'i'):
        warn = FutureWarning
    if warn_copy_on_write:
        with tm.assert_cow_warning(raise_on_extra_warnings=warn is not None):
            indexer_si(subset)[indexer] = 0
    else:
        with tm.assert_produces_warning(warn, match=msg):
            indexer_si(subset)[indexer] = 0
    expected = Series([0, 0, 3], index=['a', 'b', 'c'])
    tm.assert_series_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_series_equal(s, s_orig)
    else:
        tm.assert_series_equal(s, expected)