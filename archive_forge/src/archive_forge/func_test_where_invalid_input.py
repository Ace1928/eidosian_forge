import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('cond', [[1, 0, 1], Series([2, 5, 7]), ['True', 'False', 'True'], [Timestamp('2017-01-01'), pd.NaT, Timestamp('2017-01-02')]])
def test_where_invalid_input(cond):
    s = Series([1, 2, 3])
    msg = 'Boolean array expected for the condition'
    with pytest.raises(ValueError, match=msg):
        s.where(cond)
    msg = 'Array conditional must be same shape as self'
    with pytest.raises(ValueError, match=msg):
        s.where([True])