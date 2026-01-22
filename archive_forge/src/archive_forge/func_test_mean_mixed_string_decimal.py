from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_mean_mixed_string_decimal(self):
    d = [{'A': 2, 'B': None, 'C': Decimal('628.00')}, {'A': 1, 'B': None, 'C': Decimal('383.00')}, {'A': 3, 'B': None, 'C': Decimal('651.00')}, {'A': 2, 'B': None, 'C': Decimal('575.00')}, {'A': 4, 'B': None, 'C': Decimal('1114.00')}, {'A': 1, 'B': 'TEST', 'C': Decimal('241.00')}, {'A': 2, 'B': None, 'C': Decimal('572.00')}, {'A': 4, 'B': None, 'C': Decimal('609.00')}, {'A': 3, 'B': None, 'C': Decimal('820.00')}, {'A': 5, 'B': None, 'C': Decimal('1223.00')}]
    df = DataFrame(d)
    with pytest.raises(TypeError, match='unsupported operand type|does not support'):
        df.mean()
    result = df[['A', 'C']].mean()
    expected = Series([2.7, 681.6], index=['A', 'C'], dtype=object)
    tm.assert_series_equal(result, expected)