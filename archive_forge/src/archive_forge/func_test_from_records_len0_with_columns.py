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
def test_from_records_len0_with_columns(self):
    result = DataFrame.from_records([], index='foo', columns=['foo', 'bar'])
    expected = Index(['bar'])
    assert len(result) == 0
    assert result.index.name == 'foo'
    tm.assert_index_equal(result.columns, expected)