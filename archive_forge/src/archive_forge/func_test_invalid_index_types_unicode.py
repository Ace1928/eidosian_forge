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
@pytest.mark.skipif(is_platform_windows(), reason='see gh-10822: Windows issue')
def test_invalid_index_types_unicode():
    msg = 'Unknown datetime string format'
    with pytest.raises(ValueError, match=msg):
        frequencies.infer_freq(Index(['ZqgszYBfuL']))