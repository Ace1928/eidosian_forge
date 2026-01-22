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
@pytest.mark.parametrize('count', range(1, 5))
def test_infer_freq_delta(base_delta_code_pair, count):
    b = Timestamp(datetime.now())
    base_delta, code = base_delta_code_pair
    inc = base_delta * count
    index = DatetimeIndex([b + inc * j for j in range(3)])
    exp_freq = f'{count:d}{code}' if count > 1 else code
    assert frequencies.infer_freq(index) == exp_freq