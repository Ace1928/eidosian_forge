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
def test_order_of_appearance_dt64(self, unit):
    ser = Series([Timestamp('20160101'), Timestamp('20160101')]).dt.as_unit(unit)
    result = pd.unique(ser)
    expected = np.array(['2016-01-01T00:00:00.000000000'], dtype=f'M8[{unit}]')
    tm.assert_numpy_array_equal(result, expected)