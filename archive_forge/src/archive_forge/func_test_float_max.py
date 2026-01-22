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
def test_float_max(self, float_numpy_dtype):
    klass = np.dtype(float_numpy_dtype).type
    num = klass(np.finfo(float_numpy_dtype).max / 10)
    tm.assert_almost_equal(klass(ujson.ujson_loads(ujson.ujson_dumps(num, double_precision=15))), num)