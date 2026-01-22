import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_int_max(self, any_int_numpy_dtype):
    if any_int_numpy_dtype in ('int64', 'uint64') and (not IS64):
        pytest.skip('Cannot test 64-bit integer on 32-bit platform')
    klass = np.dtype(any_int_numpy_dtype).type
    if any_int_numpy_dtype == 'uint64':
        num = np.iinfo('int64').max
    else:
        num = np.iinfo(any_int_numpy_dtype).max
    assert klass(ujson.ujson_loads(ujson.ujson_dumps(num))) == num