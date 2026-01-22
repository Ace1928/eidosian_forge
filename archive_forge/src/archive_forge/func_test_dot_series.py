import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dot_series(self, obj, other, expected):
    result = obj.dot(other['1'])
    self.reduced_dim_assert(result, expected['1'])