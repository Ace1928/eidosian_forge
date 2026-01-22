import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_query_numexpr(self, df, expected1, expected2):
    if NUMEXPR_INSTALLED:
        result = df.query('A>0', engine='numexpr')
        tm.assert_frame_equal(result, expected1)
        result = df.eval('A+1', engine='numexpr')
        tm.assert_series_equal(result, expected2, check_names=False)
    else:
        msg = "'numexpr' is not installed or an unsupported version. Cannot use engine='numexpr' for query/eval if 'numexpr' is not installed"
        with pytest.raises(ImportError, match=msg):
            df.query('A>0', engine='numexpr')
        with pytest.raises(ImportError, match=msg):
            df.eval('A+1', engine='numexpr')