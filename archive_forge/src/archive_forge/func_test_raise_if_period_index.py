from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
def test_raise_if_period_index():
    index = period_range(start='1/1/1990', periods=20, freq='M')
    msg = 'Check the `freq` attribute instead of using infer_freq'
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(index)