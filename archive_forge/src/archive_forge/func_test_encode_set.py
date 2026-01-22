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
def test_encode_set(self):
    s = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    enc = ujson.ujson_dumps(s)
    dec = ujson.ujson_loads(enc)
    for v in dec:
        assert v in s