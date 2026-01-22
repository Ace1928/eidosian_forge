from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_by_series(self):
    ser = Series(range(5), name='foo')
    renamer = Series({1: 10, 2: 20})
    result = ser.rename(renamer)
    expected = Series(range(5), index=[0, 10, 20, 3, 4], name='foo')
    tm.assert_series_equal(result, expected)