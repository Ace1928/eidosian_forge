import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
def test_delta_to_nanoseconds_td64_MY_raises():
    msg = 'delta_to_nanoseconds does not support Y or M units, as their duration in nanoseconds is ambiguous'
    td = np.timedelta64(1234, 'Y')
    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)
    td = np.timedelta64(1234, 'M')
    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)