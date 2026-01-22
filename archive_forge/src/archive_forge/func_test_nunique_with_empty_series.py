import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nunique_with_empty_series():
    data = Series(name='name', dtype=object)
    result = data.groupby(level=0).nunique()
    expected = Series(name='name', dtype='int64')
    tm.assert_series_equal(result, expected)