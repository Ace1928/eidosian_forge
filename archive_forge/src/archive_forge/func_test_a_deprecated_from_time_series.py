import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq,freq_depr', [('2Y', '2A'), ('2Y', '2a'), ('2Y-AUG', '2A-AUG'), ('2Y-AUG', '2A-aug')])
def test_a_deprecated_from_time_series(self, freq, freq_depr):
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version. Please use '{freq[1:]}' instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        period_range(freq=freq_depr, start='1/1/2001', end='12/1/2009')