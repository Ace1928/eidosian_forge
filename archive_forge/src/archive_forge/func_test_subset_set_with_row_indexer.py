import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
from pandas.core.dtypes.common import is_float_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('indexer', [slice(0, 2), np.array([True, True, False]), np.array([0, 1])], ids=['slice', 'mask', 'array'])
def test_subset_set_with_row_indexer(backend, indexer_si, indexer, using_copy_on_write, warn_copy_on_write):
    _, DataFrame, _ = backend
    df = DataFrame({'a': [1, 2, 3, 4], 'b': [4, 5, 6, 7], 'c': [0.1, 0.2, 0.3, 0.4]})
    df_orig = df.copy()
    subset = df[1:4]
    if indexer_si is tm.setitem and isinstance(indexer, np.ndarray) and (indexer.dtype == 'int'):
        pytest.skip('setitem with labels selects on columns')
    if using_copy_on_write:
        indexer_si(subset)[indexer] = 0
    elif warn_copy_on_write:
        with tm.assert_cow_warning():
            indexer_si(subset)[indexer] = 0
    else:
        warn = SettingWithCopyWarning if indexer_si is tm.setitem else None
        with pd.option_context('chained_assignment', 'warn'):
            with tm.assert_produces_warning(warn):
                indexer_si(subset)[indexer] = 0
    expected = DataFrame({'a': [0, 0, 4], 'b': [0, 0, 7], 'c': [0.0, 0.0, 0.4]}, index=range(1, 4))
    tm.assert_frame_equal(subset, expected)
    if using_copy_on_write:
        tm.assert_frame_equal(df, df_orig)
    else:
        df_orig[1:3] = 0
        tm.assert_frame_equal(df, df_orig)