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
def test_encode_unicode_error(self):
    string = "'\udac0'"
    msg = "'utf-8' codec can't encode character '\\\\udac0' in position 1: surrogates not allowed"
    with pytest.raises(UnicodeEncodeError, match=msg):
        ujson.ujson_dumps([string])