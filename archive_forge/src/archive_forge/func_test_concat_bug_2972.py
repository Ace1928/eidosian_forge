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
def test_concat_bug_2972(self):
    ts0 = Series(np.zeros(5))
    ts1 = Series(np.ones(5))
    ts0.name = ts1.name = 'same name'
    result = concat([ts0, ts1], axis=1)
    expected = DataFrame({0: ts0, 1: ts1})
    expected.columns = ['same name', 'same name']
    tm.assert_frame_equal(result, expected)