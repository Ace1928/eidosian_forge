import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_invalid_window_non_int(self, regular):
    msg = 'passed window foobar is not compatible with a datetimelike index'
    with pytest.raises(ValueError, match=msg):
        regular.rolling(window='foobar')
    msg = 'window must be an integer'
    with pytest.raises(ValueError, match=msg):
        regular.reset_index().rolling(window='foobar')