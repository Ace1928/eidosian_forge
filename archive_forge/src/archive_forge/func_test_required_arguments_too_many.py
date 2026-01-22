import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_required_arguments_too_many(self):
    msg = 'Of the three parameters: start, end, and periods, exactly two must be specified'
    with pytest.raises(ValueError, match=msg):
        period_range(start='2017Q1', end='2018Q1', periods=8, freq='Q')