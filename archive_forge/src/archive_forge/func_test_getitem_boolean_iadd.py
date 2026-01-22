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
def test_getitem_boolean_iadd(self):
    arr = np.random.default_rng(2).standard_normal((5, 5))
    df = DataFrame(arr.copy(), columns=['A', 'B', 'C', 'D', 'E'])
    df[df < 0] += 1
    arr[arr < 0] += 1
    tm.assert_almost_equal(df.values, arr)