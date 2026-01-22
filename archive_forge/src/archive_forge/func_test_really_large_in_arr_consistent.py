import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_really_large_in_arr_consistent(large_val, signed, multiple_elts, errors):
    kwargs = {'errors': errors} if errors is not None else {}
    arr = [str(-large_val if signed else large_val)]
    if multiple_elts:
        arr.insert(0, large_val)
    if errors in (None, 'raise'):
        index = int(multiple_elts)
        msg = f'Integer out of range. at position {index}'
        with pytest.raises(ValueError, match=msg):
            to_numeric(arr, **kwargs)
    else:
        result = to_numeric(arr, **kwargs)
        if errors == 'coerce':
            expected = [float(i) for i in arr]
            exp_dtype = float
        else:
            expected = arr
            exp_dtype = object
        tm.assert_almost_equal(result, np.array(expected, dtype=exp_dtype))