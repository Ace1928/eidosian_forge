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
def test_datetime_index(self):
    date_unit = 'ns'
    rng = DatetimeIndex(list(date_range('1/1/2000', periods=20)), freq=None)
    encoded = ujson.ujson_dumps(rng, date_unit=date_unit)
    decoded = DatetimeIndex(np.array(ujson.ujson_loads(encoded)))
    tm.assert_index_equal(rng, decoded)
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    decoded = Series(ujson.ujson_loads(ujson.ujson_dumps(ts, date_unit=date_unit)))
    idx_values = decoded.index.values.astype(np.int64)
    decoded.index = DatetimeIndex(idx_values)
    tm.assert_series_equal(ts, decoded)