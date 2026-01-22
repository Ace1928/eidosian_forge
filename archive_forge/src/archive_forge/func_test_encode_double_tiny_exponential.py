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
def test_encode_double_tiny_exponential(self):
    num = 1e-40
    assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
    num = 1e-100
    assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
    num = -1e-45
    assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
    num = -1e-145
    assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))