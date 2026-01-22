from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_bug_1719(self):
    ts1 = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10))
    ts2 = ts1.copy()[::2]
    left = concat([ts1, ts2], join='outer', axis=1)
    right = concat([ts2, ts1], join='outer', axis=1)
    assert len(left) == len(right)