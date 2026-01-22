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
@pytest.mark.parametrize('long_input', [9223372036854775807, 18446744073709551615])
def test_encode_long_conversion(self, long_input):
    output = ujson.ujson_dumps(long_input)
    assert long_input == json.loads(output)
    assert output == json.dumps(long_input)
    assert long_input == ujson.ujson_loads(output)