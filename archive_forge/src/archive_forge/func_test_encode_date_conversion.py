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
def test_encode_date_conversion(self):
    date_input = datetime.date.fromtimestamp(time.time())
    output = ujson.ujson_dumps(date_input, date_unit='s')
    tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
    expected = calendar.timegm(tup)
    assert int(expected) == json.loads(output)
    assert int(expected) == ujson.ujson_loads(output)