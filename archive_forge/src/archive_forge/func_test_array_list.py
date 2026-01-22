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
def test_array_list(self):
    arr_list = ['a', [], {}, {}, [], 42, 97.8, ['a', 'b'], {'key': 'val'}]
    arr = np.array(arr_list, dtype=object)
    result = np.array(ujson.ujson_loads(ujson.ujson_dumps(arr)), dtype=object)
    tm.assert_numpy_array_equal(result, arr)