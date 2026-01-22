import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_mixed_float(self, mixed_float_frame):
    mf = mixed_float_frame.reindex(columns=['A', 'B', 'D'])
    mf.loc[mf.index[-10:], 'A'] = np.nan
    result = mf.fillna(value=0)
    _check_mixed_float(result, dtype={'C': None})
    msg = "DataFrame.fillna with 'method' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = mf.fillna(method='pad')
    _check_mixed_float(result, dtype={'C': None})