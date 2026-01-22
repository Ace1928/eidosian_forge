from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('tz', [pytz.timezone('US/Eastern'), gettz('US/Eastern')])
@pytest.mark.parametrize('use_str', [True, False])
@pytest.mark.parametrize('box_cls', [Timestamp, DatetimeIndex])
def test_dti_ambiguous_matches_timestamp(self, tz, use_str, box_cls, request):
    dtstr = '2013-11-03 01:59:59.999999'
    item = dtstr
    if not use_str:
        item = Timestamp(dtstr).to_pydatetime()
    if box_cls is not Timestamp:
        item = [item]
    if not use_str and isinstance(tz, dateutil.tz.tzfile):
        mark = pytest.mark.xfail(reason='We implicitly get fold=0.')
        request.applymarker(mark)
    with pytest.raises(pytz.AmbiguousTimeError, match=dtstr):
        box_cls(item, tz=tz)