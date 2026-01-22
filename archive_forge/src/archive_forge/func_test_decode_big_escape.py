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
def test_decode_big_escape(self):
    for _ in range(10):
        base = 'å'.encode()
        quote = b'"'
        escape_input = quote + base * 1024 * 1024 * 2 + quote
        ujson.ujson_loads(escape_input)