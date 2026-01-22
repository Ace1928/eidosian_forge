import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('order', ['F', 'C'])
def test_ravel_read_only(using_copy_on_write, order):
    ser = Series([1, 2, 3])
    with tm.assert_produces_warning(FutureWarning, match='is deprecated'):
        arr = ser.ravel(order=order)
    if using_copy_on_write:
        assert arr.flags.writeable is False
    assert np.shares_memory(get_array(ser), arr)