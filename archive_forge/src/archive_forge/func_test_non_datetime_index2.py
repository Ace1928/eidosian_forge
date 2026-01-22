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
def test_non_datetime_index2():
    rng = DatetimeIndex(['1/31/2000', '1/31/2001', '1/31/2002'])
    vals = rng.to_pydatetime()
    result = frequencies.infer_freq(vals)
    assert result == rng.inferred_freq