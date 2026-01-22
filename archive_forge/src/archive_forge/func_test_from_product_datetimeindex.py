from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_datetimeindex():
    dt_index = date_range('2000-01-01', periods=2)
    mi = MultiIndex.from_product([[1, 2], dt_index])
    etalon = construct_1d_object_array_from_listlike([(1, Timestamp('2000-01-01')), (1, Timestamp('2000-01-02')), (2, Timestamp('2000-01-01')), (2, Timestamp('2000-01-02'))])
    tm.assert_numpy_array_equal(mi.values, etalon)