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
@pytest.mark.parametrize('bigNum', [sys.maxsize + 1, -(sys.maxsize + 2)])
def test_to_json_large_numbers(self, bigNum):
    series = Series(bigNum, dtype=object, index=['articleId'])
    json = series.to_json()
    expected = '{"articleId":' + str(bigNum) + '}'
    assert json == expected
    df = DataFrame(bigNum, dtype=object, index=['articleId'], columns=[0])
    json = df.to_json()
    expected = '{"0":{"articleId":' + str(bigNum) + '}}'
    assert json == expected