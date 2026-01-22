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
def test_constructor_wrong_precision_raises(self):
    dti = DatetimeIndex(['2000'], dtype='datetime64[us]')
    assert dti.dtype == 'M8[us]'
    assert dti[0] == Timestamp(2000, 1, 1)