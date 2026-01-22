from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_dti_intersection(self):
    rng = date_range('1/1/2011', periods=100, freq='h', tz='utc')
    left = rng[10:90][::-1]
    right = rng[20:80][::-1]
    assert left.tz == rng.tz
    result = left.intersection(right)
    assert result.tz == left.tz