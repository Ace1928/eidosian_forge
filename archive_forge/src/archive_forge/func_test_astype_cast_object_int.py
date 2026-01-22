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
def test_astype_cast_object_int(self):
    arr = Series(['1', '2', '3', '4'], dtype=object)
    result = arr.astype(int)
    tm.assert_series_equal(result, Series(np.arange(1, 5)))