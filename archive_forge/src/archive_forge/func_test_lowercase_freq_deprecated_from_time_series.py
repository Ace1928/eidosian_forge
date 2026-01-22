import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq_depr', ['2m', '2q-sep', '2y', '2w'])
def test_lowercase_freq_deprecated_from_time_series(self, freq_depr):
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version. Please use '{freq_depr.upper()[1:]}' instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        period_range(freq=freq_depr, start='1/1/2001', end='12/1/2009')