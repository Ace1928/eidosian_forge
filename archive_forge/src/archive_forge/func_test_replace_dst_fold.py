from datetime import datetime
from dateutil.tz import gettz
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
import pandas.util._test_decorators as td
import pandas._testing as tm
@pytest.mark.parametrize('fold', [0, 1])
@pytest.mark.parametrize('tz', ['dateutil/Europe/London', 'Europe/London'])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_replace_dst_fold(self, fold, tz, unit):
    d = datetime(2019, 10, 27, 2, 30)
    ts = Timestamp(d, tz=tz).as_unit(unit)
    result = ts.replace(hour=1, fold=fold)
    expected = Timestamp(datetime(2019, 10, 27, 1, 30)).tz_localize(tz, ambiguous=not fold)
    assert result == expected
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value