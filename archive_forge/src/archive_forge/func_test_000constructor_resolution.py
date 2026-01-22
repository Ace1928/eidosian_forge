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
def test_000constructor_resolution(self):
    t1 = Timestamp(1352934390 * 1000000000 + 1000000 + 1000 + 1)
    idx = DatetimeIndex([t1])
    assert idx.nanosecond[0] == t1.nanosecond