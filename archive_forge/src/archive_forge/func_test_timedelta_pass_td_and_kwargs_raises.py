from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_timedelta_pass_td_and_kwargs_raises():
    td = Timedelta(days=1)
    msg = "Cannot pass both a Timedelta input and timedelta keyword arguments, got \\['days'\\]"
    with pytest.raises(ValueError, match=msg):
        Timedelta(td, days=2)