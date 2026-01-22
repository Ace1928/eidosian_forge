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
@pytest.fixture
def datetime_series(self):
    ser = Series(1.1 * np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    ser.index = ser.index._with_freq(None)
    return ser