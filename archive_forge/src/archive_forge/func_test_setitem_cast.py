from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_setitem_cast(self, float_frame):
    float_frame['D'] = float_frame['D'].astype('i8')
    assert float_frame['D'].dtype == np.int64
    float_frame['B'] = 0
    assert float_frame['B'].dtype == np.int64
    float_frame['B'] = np.arange(len(float_frame))
    assert issubclass(float_frame['B'].dtype.type, np.integer)
    float_frame['foo'] = 'bar'
    float_frame['foo'] = 0
    assert float_frame['foo'].dtype == np.int64
    float_frame['foo'] = 'bar'
    float_frame['foo'] = 2.5
    assert float_frame['foo'].dtype == np.float64
    float_frame['something'] = 0
    assert float_frame['something'].dtype == np.int64
    float_frame['something'] = 2
    assert float_frame['something'].dtype == np.int64
    float_frame['something'] = 2.5
    assert float_frame['something'].dtype == np.float64