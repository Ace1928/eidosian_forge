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
@pytest.mark.parametrize('int_exp', ['1337E40', '1.337E40', '1337E+9', '1.337e+40', '1.337E-4'])
def test_decode_numeric_int_exp(self, int_exp):
    assert ujson.ujson_loads(int_exp) == json.loads(int_exp)