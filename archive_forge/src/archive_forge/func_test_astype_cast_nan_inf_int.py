import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [np.int32, np.int64])
@pytest.mark.parametrize('val', [np.nan, np.inf])
def test_astype_cast_nan_inf_int(self, val, dtype):
    msg = 'Cannot convert non-finite values \\(NA or inf\\) to integer'
    df = DataFrame([val])
    with pytest.raises(ValueError, match=msg):
        df.astype(dtype)