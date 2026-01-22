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
def test_concat_keys_with_none(self):
    df0 = DataFrame([[10, 20, 30], [10, 20, 30], [10, 20, 30]])
    result = concat({'a': None, 'b': df0, 'c': df0[:2], 'd': df0[:1], 'e': df0})
    expected = concat({'b': df0, 'c': df0[:2], 'd': df0[:1], 'e': df0})
    tm.assert_frame_equal(result, expected)
    result = concat([None, df0, df0[:2], df0[:1], df0], keys=['a', 'b', 'c', 'd', 'e'])
    expected = concat([df0, df0[:2], df0[:1], df0], keys=['b', 'c', 'd', 'e'])
    tm.assert_frame_equal(result, expected)