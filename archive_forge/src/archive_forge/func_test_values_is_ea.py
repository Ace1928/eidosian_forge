import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_values_is_ea(using_copy_on_write):
    df = DataFrame({'a': date_range('2012-01-01', periods=3)})
    arr = np.asarray(df)
    if using_copy_on_write:
        assert arr.flags.writeable is False
    else:
        assert arr.flags.writeable is True