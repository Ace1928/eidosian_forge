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
@pytest.mark.parametrize('invalid_val', [20, -1, '9', None])
def test_invalid_double_precision(self, invalid_val):
    double_input = 30.123456789012344
    expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
    msg = "Invalid value '.*' for option 'double_precision', max is '15'|an integer is required \\(got type |object cannot be interpreted as an integer"
    with pytest.raises(expected_exception, match=msg):
        ujson.ujson_dumps(double_input, double_precision=invalid_val)