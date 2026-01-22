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
@pytest.mark.parametrize('index', [None, [1, 2], [1.0, 2.0], ['a', 'b'], ['1', '2'], ['1.', '2.']])
@pytest.mark.parametrize('columns', [['a', 'b'], ['1', '2'], ['1.', '2.']])
def test_from_json_to_json_table_index_and_columns(self, index, columns):
    expected = DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
    dfjson = expected.to_json(orient='table')
    result = read_json(StringIO(dfjson), orient='table')
    tm.assert_frame_equal(result, expected)