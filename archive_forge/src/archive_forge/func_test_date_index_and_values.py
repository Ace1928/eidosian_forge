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
@pytest.mark.parametrize('date_format', ['epoch', 'iso'])
@pytest.mark.parametrize('as_object', [True, False])
@pytest.mark.parametrize('date_typ', [datetime.date, datetime.datetime, Timestamp])
def test_date_index_and_values(self, date_format, as_object, date_typ):
    data = [date_typ(year=2020, month=1, day=1), pd.NaT]
    if as_object:
        data.append('a')
    ser = Series(data, index=data)
    result = ser.to_json(date_format=date_format)
    if date_format == 'epoch':
        expected = '{"1577836800000":1577836800000,"null":null}'
    else:
        expected = '{"2020-01-01T00:00:00.000":"2020-01-01T00:00:00.000","null":null}'
    if as_object:
        expected = expected.replace('}', ',"a":"a"}')
    assert result == expected