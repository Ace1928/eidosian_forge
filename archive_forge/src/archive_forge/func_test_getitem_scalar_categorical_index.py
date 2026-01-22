from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_scalar_categorical_index(self):
    cats = Categorical([Timestamp('12-31-1999'), Timestamp('12-31-2000')])
    ser = Series([1, 2], index=cats)
    expected = ser.iloc[0]
    result = ser[cats[0]]
    assert result == expected