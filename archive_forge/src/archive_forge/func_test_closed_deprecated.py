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
def test_closed_deprecated(self):
    msg = "The 'closed' keyword"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        DatetimeIndex([], closed=True)