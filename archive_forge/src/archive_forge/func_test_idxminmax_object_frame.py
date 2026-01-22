from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxminmax_object_frame(self):
    df = DataFrame([['zimm', 2.5], ['biff', 1.0], ['bid', 12.0]])
    res = df.idxmax()
    exp = Series([0, 2])
    tm.assert_series_equal(res, exp)