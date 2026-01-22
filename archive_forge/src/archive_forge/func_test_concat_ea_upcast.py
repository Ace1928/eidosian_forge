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
def test_concat_ea_upcast():
    df1 = DataFrame(['a'], dtype='string')
    df2 = DataFrame([1], dtype='Int64')
    result = concat([df1, df2])
    expected = DataFrame(['a', 1], index=[0, 0])
    tm.assert_frame_equal(result, expected)