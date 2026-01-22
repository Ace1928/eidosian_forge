from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_attribute(self):
    df = DataFrame({'x': [1, 2, 3]})
    df.y = 2
    df['y'] = [2, 4, 6]
    df.y = 5
    assert df.y == 5
    tm.assert_series_equal(df['y'], Series([2, 4, 6], name='y'))