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
def test_frame_from_records_utc(self):
    rec = {'datum': 1.5, 'begin_time': datetime(2006, 4, 27, tzinfo=pytz.utc)}
    DataFrame.from_records([rec], index='begin_time')