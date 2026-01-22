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
def test_unary_non_nano(self, td, unit):
    assert abs(td)._creso == unit
    assert (-td)._creso == unit
    assert (+td)._creso == unit