from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_timedelta_conversions(self):
    assert Timedelta(timedelta(seconds=1)) == np.timedelta64(1, 's').astype('m8[ns]')
    assert Timedelta(timedelta(microseconds=1)) == np.timedelta64(1, 'us').astype('m8[ns]')
    assert Timedelta(timedelta(days=1)) == np.timedelta64(1, 'D').astype('m8[ns]')