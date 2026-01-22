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
def test_encode_dict_conversion(self):
    dict_input = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4}
    output = ujson.ujson_dumps(dict_input)
    assert dict_input == json.loads(output)
    assert dict_input == ujson.ujson_loads(output)