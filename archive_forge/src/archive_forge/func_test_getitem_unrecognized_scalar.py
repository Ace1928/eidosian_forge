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
def test_getitem_unrecognized_scalar(self):
    ser = Series([1, 2], index=[np.dtype('O'), np.dtype('i8')])
    key = ser.index[1]
    result = ser[key]
    assert result == 2