import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_mismatched_start_end_freq_raises(self):
    depr_msg = 'Period with BDay freq is deprecated'
    msg = "'w' is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        end_w = Period('2006-12-31', '1w')
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        start_b = Period('02-Apr-2005', 'B')
        end_b = Period('2005-05-01', 'B')
    msg = 'start and end must have same freq'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            period_range(start=start_b, end=end_w)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        period_range(start=start_b, end=end_b)