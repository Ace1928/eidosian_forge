import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@td.skip_array_manager_not_yet_implemented
def test_fillna_on_column_view(self, using_copy_on_write):
    arr = np.full((40, 50), np.nan)
    df = DataFrame(arr, copy=False)
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df[0].fillna(-1, inplace=True)
        assert np.isnan(arr[:, 0]).all()
    else:
        with tm.assert_produces_warning(FutureWarning, match='inplace method'):
            df[0].fillna(-1, inplace=True)
        assert (arr[:, 0] == -1).all()
    assert len(df._mgr.arrays) == 1
    assert np.shares_memory(df.values, arr)