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
@pytest.mark.parametrize('broken_json', ['[', '{', ']', '}'])
def test_decode_broken_json(self, broken_json):
    msg = 'Expected object or value'
    with pytest.raises(ValueError, match=msg):
        ujson.ujson_loads(broken_json)