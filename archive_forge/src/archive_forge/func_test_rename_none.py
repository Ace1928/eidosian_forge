from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_none(self):
    ser = Series([1, 2], name='foo')
    result = ser.rename(None)
    expected = Series([1, 2])
    tm.assert_series_equal(result, expected)