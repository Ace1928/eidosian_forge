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
@td.skip_if_no('pyarrow')
def test_to_json_ea_null():
    df = DataFrame({'a': Series([1, NA], dtype='int64[pyarrow]'), 'b': Series([2, NA], dtype='Int64')})
    result = df.to_json(orient='records', lines=True)
    expected = '{"a":1,"b":2}\n{"a":null,"b":null}\n'
    assert result == expected