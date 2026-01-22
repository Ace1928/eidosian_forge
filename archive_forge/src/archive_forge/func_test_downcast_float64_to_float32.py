import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_downcast_float64_to_float32():
    series = Series([16777217.0, np.finfo(np.float64).max, np.nan], dtype=np.float64)
    result = to_numeric(series, downcast='float')
    assert series.dtype == result.dtype