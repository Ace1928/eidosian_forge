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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='incorrect na conversion')
@pytest.mark.parametrize('orient', ['split', 'records', 'index', 'columns'])
def test_to_json_from_json_columns_dtypes(self, orient):
    expected = DataFrame.from_dict({'Integer': Series([1, 2, 3], dtype='int64'), 'Float': Series([None, 2.0, 3.0], dtype='float64'), 'Object': Series([None, '', 'c'], dtype='object'), 'Bool': Series([True, False, True], dtype='bool'), 'Category': Series(['a', 'b', None], dtype='category'), 'Datetime': Series(['2020-01-01', None, '2020-01-03'], dtype='datetime64[ns]')})
    dfjson = expected.to_json(orient=orient)
    result = read_json(StringIO(dfjson), orient=orient, dtype={'Integer': 'int64', 'Float': 'float64', 'Object': 'object', 'Bool': 'bool', 'Category': 'category', 'Datetime': 'datetime64[ns]'})
    tm.assert_frame_equal(result, expected)