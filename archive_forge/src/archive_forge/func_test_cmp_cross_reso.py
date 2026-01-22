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
def test_cmp_cross_reso(self, td):
    other = Timedelta(days=106751, unit='ns')
    assert other < td
    assert td > other
    assert not other == td
    assert td != other