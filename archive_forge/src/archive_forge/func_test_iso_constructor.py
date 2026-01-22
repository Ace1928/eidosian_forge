from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('fmt,exp', [('P6DT0H50M3.010010012S', Timedelta(days=6, minutes=50, seconds=3, milliseconds=10, microseconds=10, nanoseconds=12)), ('P-6DT0H50M3.010010012S', Timedelta(days=-6, minutes=50, seconds=3, milliseconds=10, microseconds=10, nanoseconds=12)), ('P4DT12H30M5S', Timedelta(days=4, hours=12, minutes=30, seconds=5)), ('P0DT0H0M0.000000123S', Timedelta(nanoseconds=123)), ('P0DT0H0M0.00001S', Timedelta(microseconds=10)), ('P0DT0H0M0.001S', Timedelta(milliseconds=1)), ('P0DT0H1M0S', Timedelta(minutes=1)), ('P1DT25H61M61S', Timedelta(days=1, hours=25, minutes=61, seconds=61)), ('PT1S', Timedelta(seconds=1)), ('PT0S', Timedelta(seconds=0)), ('P1WT0S', Timedelta(days=7, seconds=0)), ('P1D', Timedelta(days=1)), ('P1DT1H', Timedelta(days=1, hours=1)), ('P1W', Timedelta(days=7)), ('PT300S', Timedelta(seconds=300)), ('P1DT0H0M00000000000S', Timedelta(days=1)), ('PT-6H3M', Timedelta(hours=-6, minutes=3)), ('-PT6H3M', Timedelta(hours=-6, minutes=-3)), ('-PT-6H+3M', Timedelta(hours=6, minutes=-3))])
def test_iso_constructor(fmt, exp):
    assert Timedelta(fmt) == exp