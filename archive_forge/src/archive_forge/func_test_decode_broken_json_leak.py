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
@pytest.mark.parametrize('broken_json, err_msg', [('{{1337:""}}', "Key name of object must be 'string' when decoding 'object'"), ('{{"key":"}', 'Unmatched \'\'"\' when when decoding \'string\''), ('[[[true', 'Unexpected character found when decoding array value (2)')])
def test_decode_broken_json_leak(self, broken_json, err_msg):
    for _ in range(1000):
        with pytest.raises(ValueError, match=re.escape(err_msg)):
            ujson.ujson_loads(broken_json)