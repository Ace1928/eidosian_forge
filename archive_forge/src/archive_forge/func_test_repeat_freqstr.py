import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('use_numpy', [True, False])
@pytest.mark.parametrize('index', [period_range('2000-01-01', periods=3, freq='D'), period_range('2001-01-01', periods=3, freq='2D'), PeriodIndex(['2001-01', 'NaT', '2003-01'], freq='M')])
def test_repeat_freqstr(self, index, use_numpy):
    expected = PeriodIndex([per for per in index for _ in range(3)])
    result = np.repeat(index, 3) if use_numpy else index.repeat(3)
    tm.assert_index_equal(result, expected)
    assert result.freqstr == index.freqstr