from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_multifunc_sum_bug():
    x = DataFrame(np.arange(9).reshape(3, 3))
    x['test'] = 0
    x['fl'] = [1.3, 1.5, 1.6]
    grouped = x.groupby('test')
    result = grouped.agg({'fl': 'sum', 2: 'size'})
    assert result['fl'].dtype == np.float64