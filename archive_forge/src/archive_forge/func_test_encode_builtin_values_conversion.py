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
@pytest.mark.parametrize('builtin_value', [None, True, False])
def test_encode_builtin_values_conversion(self, builtin_value):
    output = ujson.ujson_dumps(builtin_value)
    assert builtin_value == json.loads(output)
    assert output == json.dumps(builtin_value)
    assert builtin_value == ujson.ujson_loads(output)