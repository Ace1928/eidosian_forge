from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('data_type', [np.dtype(f'f{width}') for width in [4, 8]] + [np.dtype(f'{sign}{width}') for width in [1, 2, 4, 8] for sign in 'ui'])
def test_rolling_min_max_numeric_types(data_type):
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).max()
    assert result.dtypes[0] == np.dtype('f8')
    result = DataFrame(np.arange(20, dtype=data_type)).rolling(window=5).min()
    assert result.dtypes[0] == np.dtype('f8')