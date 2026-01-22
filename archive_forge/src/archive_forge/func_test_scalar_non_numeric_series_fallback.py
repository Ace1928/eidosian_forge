import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [Index(list('abcde')), Index(list('abcde'), dtype='category'), date_range('2020-01-01', periods=5), timedelta_range('1 day', periods=5), period_range('2020-01-01', periods=5)])
def test_scalar_non_numeric_series_fallback(self, index):
    s = Series(np.arange(len(index)), index=index)
    msg = 'Series.__getitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s[3]
    with pytest.raises(KeyError, match='^3.0$'):
        s[3.0]