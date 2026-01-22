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
def test_convert_pandas_type_to_json_field_int(self, index_or_series):
    kind = index_or_series
    data = [1, 2, 3]
    result = convert_pandas_type_to_json_field(kind(data, name='name'))
    expected = {'name': 'name', 'type': 'integer'}
    assert result == expected