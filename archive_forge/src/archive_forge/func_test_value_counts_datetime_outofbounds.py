from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('dtype', [object, 'M8[us]'])
def test_value_counts_datetime_outofbounds(self, dtype):
    ser = Series([datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1), datetime(3000, 1, 1), datetime(3000, 1, 1)], dtype=dtype)
    res = ser.value_counts()
    exp_index = Index([datetime(3000, 1, 1), datetime(5000, 1, 1), datetime(6000, 1, 1)], dtype=dtype)
    exp = Series([3, 2, 1], index=exp_index, name='count')
    tm.assert_series_equal(res, exp)