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
def test_dataframe_nested(self, orient):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], index=['a', 'b'], columns=['x', 'y', 'z'])
    nested = {'df1': df, 'df2': df.copy()}
    kwargs = {} if orient is None else {'orient': orient}
    exp = {'df1': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs)), 'df2': ujson.ujson_loads(ujson.ujson_dumps(df, **kwargs))}
    assert ujson.ujson_loads(ujson.ujson_dumps(nested, **kwargs)) == exp