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
def test_concat_empty_and_non_empty_frame_regression():
    df1 = DataFrame({'foo': [1]})
    df2 = DataFrame({'foo': []})
    expected = DataFrame({'foo': [1.0]})
    result = concat([df1, df2])
    tm.assert_frame_equal(result, expected)