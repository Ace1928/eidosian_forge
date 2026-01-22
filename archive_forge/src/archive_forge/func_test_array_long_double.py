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
def test_array_long_double(self):
    msg = re.compile('1234.5.* \\(numpy-scalar\\) is not JSON serializable at the moment')
    with pytest.raises(TypeError, match=msg):
        ujson.ujson_dumps(np.longdouble(1234.5))