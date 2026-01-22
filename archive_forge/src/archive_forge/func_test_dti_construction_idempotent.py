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
def test_dti_construction_idempotent(self, unit):
    rng = date_range('03/12/2012 00:00', periods=10, freq='W-FRI', tz='US/Eastern', unit=unit)
    rng2 = DatetimeIndex(data=rng, tz='US/Eastern')
    tm.assert_index_equal(rng, rng2)