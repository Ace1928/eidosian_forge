import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_aligns(self, obj, other, expected):
    other2 = other.iloc[::-1]
    result = obj.dot(other2)
    tm.assert_equal(result, expected)