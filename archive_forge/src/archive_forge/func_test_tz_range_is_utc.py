import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.parametrize('tz_range', [date_range('2013-01-01 05:00:00Z', periods=2), date_range('2013-01-01 00:00:00', periods=2, tz='US/Eastern'), date_range('2013-01-01 00:00:00-0500', periods=2)])
def test_tz_range_is_utc(self, tz_range):
    exp = '["2013-01-01T05:00:00.000Z","2013-01-02T05:00:00.000Z"]'
    dfexp = '{"DT":{"0":"2013-01-01T05:00:00.000Z","1":"2013-01-02T05:00:00.000Z"}}'
    assert ujson_dumps(tz_range, iso_dates=True) == exp
    dti = DatetimeIndex(tz_range)
    assert ujson_dumps(dti, iso_dates=True) == exp
    assert ujson_dumps(dti.astype(object), iso_dates=True) == exp
    df = DataFrame({'DT': dti})
    result = ujson_dumps(df, iso_dates=True)
    assert result == dfexp
    assert ujson_dumps(df.astype({'DT': object}), iso_dates=True)