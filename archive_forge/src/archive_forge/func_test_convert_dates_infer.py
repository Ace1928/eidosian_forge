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
@pytest.mark.parametrize('infer_word', ['trade_time', 'date', 'datetime', 'sold_at', 'modified', 'timestamp', 'timestamps'])
def test_convert_dates_infer(self, infer_word):
    data = [{'id': 1, infer_word: 1036713600000}, {'id': 2}]
    expected = DataFrame([[1, Timestamp('2002-11-08')], [2, pd.NaT]], columns=['id', infer_word])
    result = read_json(StringIO(ujson_dumps(data)))[['id', infer_word]]
    tm.assert_frame_equal(result, expected)