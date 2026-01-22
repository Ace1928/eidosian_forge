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
def test_from_records_non_tuple(self):

    class Record:

        def __init__(self, *args) -> None:
            self.args = args

        def __getitem__(self, i):
            return self.args[i]

        def __iter__(self) -> Iterator:
            return iter(self.args)
    recs = [Record(1, 2, 3), Record(4, 5, 6), Record(7, 8, 9)]
    tups = [tuple(rec) for rec in recs]
    result = DataFrame.from_records(recs)
    expected = DataFrame.from_records(tups)
    tm.assert_frame_equal(result, expected)