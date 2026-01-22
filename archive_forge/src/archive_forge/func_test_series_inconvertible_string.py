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
def test_series_inconvertible_string(using_infer_string):
    if using_infer_string:
        msg = 'cannot infer freq from'
        with pytest.raises(TypeError, match=msg):
            frequencies.infer_freq(Series(['foo', 'bar']))
    else:
        msg = 'Unknown datetime string format'
        with pytest.raises(ValueError, match=msg):
            frequencies.infer_freq(Series(['foo', 'bar']))