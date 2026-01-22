from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_unit64_float_conversion():
    df = DataFrame({'first': [1], 'second': [1], 'value': [16148277970000000000]})
    result = df.groupby(['first', 'second'])['value'].max()
    expected = Series([16148277970000000000], MultiIndex.from_product([[1], [1]], names=['first', 'second']), name='value')
    tm.assert_series_equal(result, expected)