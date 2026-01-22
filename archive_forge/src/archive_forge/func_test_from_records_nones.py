from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_nones(self):
    tuples = [(1, 2, None, 3), (1, 2, None, 3), (None, 2, 5, 3)]
    df = DataFrame.from_records(tuples, columns=['a', 'b', 'c', 'd'])
    assert np.isnan(df['c'][0])