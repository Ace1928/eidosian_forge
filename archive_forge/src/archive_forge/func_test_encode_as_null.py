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
@pytest.mark.parametrize('decoded_input', [NaT, np.datetime64('NaT'), np.nan, np.inf, -np.inf])
def test_encode_as_null(self, decoded_input):
    assert ujson.ujson_dumps(decoded_input) == 'null', 'Expected null'