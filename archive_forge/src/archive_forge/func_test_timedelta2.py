import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_timedelta2(self):
    frame = DataFrame({'a': [timedelta(days=23), timedelta(seconds=5)], 'b': [1, 2], 'c': date_range(start='20130101', periods=2)})
    data = StringIO(frame.to_json(date_unit='ns'))
    result = read_json(data)
    result['a'] = pd.to_timedelta(result.a, unit='ns')
    result['c'] = pd.to_datetime(result.c)
    tm.assert_frame_equal(frame, result)