import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_nan_idx(self):
    df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [np.nan, 2, 3, 4]})
    df = df.set_index('A')
    msg = 'Interpolation with NaNs in the index has not been implemented. Try filling those NaNs before interpolating.'
    with pytest.raises(NotImplementedError, match=msg):
        df.interpolate(method='values')