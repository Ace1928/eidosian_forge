import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_periods_requires_integer(self):
    msg = 'periods must be a number, got foo'
    with pytest.raises(TypeError, match=msg):
        period_range(start='2017Q1', periods='foo')