import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_equiv_values_dot(self, obj, other, expected):
    result = obj.dot(other)
    tm.assert_equal(result, expected)