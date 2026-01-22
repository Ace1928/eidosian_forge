import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_datetimelike_attribute_access(self):
    ser = Series({'year': 2000, 'month': 1, 'day': 10})
    assert ser.year == 2000
    assert ser.month == 1
    assert ser.day == 10