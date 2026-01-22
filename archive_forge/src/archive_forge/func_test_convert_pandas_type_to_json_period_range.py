from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
def test_convert_pandas_type_to_json_period_range(self):
    arr = pd.period_range('2016', freq='Y-DEC', periods=4)
    result = convert_pandas_type_to_json_field(arr)
    expected = {'name': 'values', 'type': 'datetime', 'freq': 'YE-DEC'}
    assert result == expected