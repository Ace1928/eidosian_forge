import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:errors='ignore' is deprecated:FutureWarning")
def test_really_large_scalar(large_val, signed, transform, errors):
    kwargs = {'errors': errors} if errors is not None else {}
    val = -large_val if signed else large_val
    val = transform(val)
    val_is_string = isinstance(val, str)
    if val_is_string and errors in (None, 'raise'):
        msg = 'Integer out of range. at position 0'
        with pytest.raises(ValueError, match=msg):
            to_numeric(val, **kwargs)
    else:
        expected = float(val) if errors == 'coerce' and val_is_string else val
        tm.assert_almost_equal(to_numeric(val, **kwargs), expected)