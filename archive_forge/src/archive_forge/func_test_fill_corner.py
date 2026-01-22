import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 0 in string")
def test_fill_corner(self, float_frame, float_string_frame):
    mf = float_string_frame
    mf.loc[mf.index[5:20], 'foo'] = np.nan
    mf.loc[mf.index[-10:], 'A'] = np.nan
    filled = float_string_frame.fillna(value=0)
    assert (filled.loc[filled.index[5:20], 'foo'] == 0).all()
    del float_string_frame['foo']
    float_frame.reindex(columns=[]).fillna(value=0)