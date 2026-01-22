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
@pytest.mark.parametrize('method', ['ceil', 'round', 'floor'])
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_round_dst_border_ambiguous(self, method, unit):
    ts = Timestamp('2017-10-29 00:00:00', tz='UTC').tz_convert('Europe/Madrid')
    ts = ts.as_unit(unit)
    result = getattr(ts, method)('H', ambiguous=True)
    assert result == ts
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
    result = getattr(ts, method)('H', ambiguous=False)
    expected = Timestamp('2017-10-29 01:00:00', tz='UTC').tz_convert('Europe/Madrid')
    assert result == expected
    assert result._creso == getattr(NpyDatetimeUnit, f'NPY_FR_{unit}').value
    result = getattr(ts, method)('H', ambiguous='NaT')
    assert result is NaT
    msg = 'Cannot infer dst time'
    with pytest.raises(pytz.AmbiguousTimeError, match=msg):
        getattr(ts, method)('H', ambiguous='raise')