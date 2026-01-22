import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_axis_unnamed_kwarg_warns(self, obj):
    new_index = list('abcd')[:len(obj)]
    expected = obj.copy()
    expected.index = new_index
    result = obj.set_axis(new_index)
    tm.assert_equal(result, expected)