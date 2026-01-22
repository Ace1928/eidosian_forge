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
def test_loads_non_str_bytes_raises(self):
    msg = "a bytes-like object is required, not 'NoneType'"
    with pytest.raises(TypeError, match=msg):
        ujson.ujson_loads(None)