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
def test_index_cast_datetime64_other_units(self):
    arr = np.arange(0, 100, 10, dtype=np.int64).view('M8[D]')
    idx = Index(arr)
    assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype('M8[ns]'))).all()