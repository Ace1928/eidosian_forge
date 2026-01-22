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
@pytest.mark.parametrize('invalid_arr', ['[31337,]', '[,31337]', '[]]', '[,]'])
def test_decode_invalid_array(self, invalid_arr):
    msg = 'Expected object or value|Trailing data|Unexpected character found when decoding array value'
    with pytest.raises(ValueError, match=msg):
        ujson.ujson_loads(invalid_arr)