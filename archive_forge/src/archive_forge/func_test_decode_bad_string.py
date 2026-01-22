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
@pytest.mark.parametrize('bad_string', ['"TESTING', '"TESTING\\"', 'tru', 'fa', 'n'])
def test_decode_bad_string(self, bad_string):
    msg = 'Unexpected character found when decoding|Unmatched \'\'"\' when when decoding \'string\''
    with pytest.raises(ValueError, match=msg):
        ujson.ujson_loads(bad_string)