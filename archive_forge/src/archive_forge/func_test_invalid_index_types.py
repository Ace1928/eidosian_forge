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
@pytest.mark.parametrize('idx', [Index(np.arange(5), dtype=np.int64), Index(np.arange(5), dtype=np.float64), period_range('2020-01-01', periods=5), RangeIndex(5)])
def test_invalid_index_types(idx):
    msg = '|'.join(['cannot infer freq from a non-convertible', 'Check the `freq` attribute instead of using infer_freq'])
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(idx)