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
def test_encode_big_escape(self):
    for _ in range(10):
        base = 'å'.encode()
        escape_input = base * 1024 * 1024 * 2
        ujson.ujson_dumps(escape_input)