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
def test_nat_converters(self):
    result = to_timedelta('nat').to_numpy()
    assert result.dtype.kind == 'M'
    assert result.astype('int64') == iNaT
    result = to_timedelta('nan').to_numpy()
    assert result.dtype.kind == 'M'
    assert result.astype('int64') == iNaT