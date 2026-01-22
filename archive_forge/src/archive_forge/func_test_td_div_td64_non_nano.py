from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_div_td64_non_nano(self):
    td = Timedelta('1 days 2 hours 3 ns')
    result = td / np.timedelta64(1, 'D')
    assert result == td._value / (86400 * 10 ** 9)
    result = td / np.timedelta64(1, 's')
    assert result == td._value / 10 ** 9
    result = td / np.timedelta64(1, 'ns')
    assert result == td._value
    td = Timedelta('1 days 2 hours 3 ns')
    result = td // np.timedelta64(1, 'D')
    assert result == 1
    result = td // np.timedelta64(1, 's')
    assert result == 93600
    result = td // np.timedelta64(1, 'ns')
    assert result == td._value