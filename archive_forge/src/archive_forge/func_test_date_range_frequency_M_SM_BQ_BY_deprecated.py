from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
@pytest.mark.parametrize('freq,freq_depr', [('2ME', '2M'), ('2SME', '2SM'), ('2BQE', '2BQ'), ('2BYE', '2BY')])
def test_date_range_frequency_M_SM_BQ_BY_deprecated(self, freq, freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    expected = date_range('1/1/2000', periods=4, freq=freq)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        result = date_range('1/1/2000', periods=4, freq=freq_depr)
    tm.assert_index_equal(result, expected)