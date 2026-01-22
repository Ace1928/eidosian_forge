from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_tz_localize_invalidates_freq(self):
    dti = date_range('2014-03-08 23:00', '2014-03-09 09:00', freq='h')
    assert dti.freq == 'h'
    result = dti.tz_localize(None)
    assert result.freq == 'h'
    result = dti.tz_localize('UTC')
    assert result.freq == 'h'
    result = dti.tz_localize('US/Eastern', nonexistent='shift_forward')
    assert result.freq is None
    assert result.inferred_freq is None
    dti2 = dti[:1]
    result = dti2.tz_localize('US/Eastern')
    assert result.freq == 'h'