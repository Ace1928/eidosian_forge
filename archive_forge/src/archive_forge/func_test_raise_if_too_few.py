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
def test_raise_if_too_few():
    index = DatetimeIndex(['12/31/1998', '1/3/1999'])
    msg = 'Need at least 3 dates to infer frequency'
    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(index)