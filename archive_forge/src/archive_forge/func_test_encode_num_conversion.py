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
@pytest.mark.parametrize('num_input', [31337, -31337, -9223372036854775808])
def test_encode_num_conversion(self, num_input):
    output = ujson.ujson_dumps(num_input)
    assert num_input == json.loads(output)
    assert output == json.dumps(num_input)
    assert num_input == ujson.ujson_loads(output)