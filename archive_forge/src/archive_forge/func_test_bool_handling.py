import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('errors,exp', [('raise', 'Unable to parse string "apple" at position 2'), ('ignore', [True, False, 'apple']), ('coerce', [1.0, 0.0, np.nan])])
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_bool_handling(errors, exp):
    ser = Series([True, False, 'apple'])
    if isinstance(exp, str):
        with pytest.raises(ValueError, match=exp):
            to_numeric(ser, errors=errors)
    else:
        result = to_numeric(ser, errors=errors)
        expected = Series(exp)
        tm.assert_series_equal(result, expected)