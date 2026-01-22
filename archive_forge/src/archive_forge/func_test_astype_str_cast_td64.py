from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_str_cast_td64(self):
    td = Series([Timedelta(1, unit='d')])
    ser = td.astype(str)
    expected = Series(['1 days'], dtype=object)
    tm.assert_series_equal(ser, expected)