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
@pytest.mark.parametrize('double_input', [math.pi, -math.pi])
def test_encode_double_conversion(self, double_input):
    output = ujson.ujson_dumps(double_input)
    assert round(double_input, 5) == round(json.loads(output), 5)
    assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)