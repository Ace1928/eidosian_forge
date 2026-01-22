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
def test_series_nested(self, orient):
    s = Series([10, 20, 30, 40, 50, 60], name='series', index=[6, 7, 8, 9, 10, 15]).sort_values()
    nested = {'s1': s, 's2': s.copy()}
    kwargs = {} if orient is None else {'orient': orient}
    exp = {'s1': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs)), 's2': ujson.ujson_loads(ujson.ujson_dumps(s, **kwargs))}
    assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp