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
@pytest.mark.parametrize('inp,exp', [({'type': 'integer'}, 'int64'), ({'type': 'number'}, 'float64'), ({'type': 'boolean'}, 'bool'), ({'type': 'duration'}, 'timedelta64'), ({'type': 'datetime'}, 'datetime64[ns]'), ({'type': 'datetime', 'tz': 'US/Hawaii'}, 'datetime64[ns, US/Hawaii]'), ({'type': 'any'}, 'object'), ({'type': 'any', 'constraints': {'enum': ['a', 'b', 'c']}, 'ordered': False}, CategoricalDtype(categories=['a', 'b', 'c'], ordered=False)), ({'type': 'any', 'constraints': {'enum': ['a', 'b', 'c']}, 'ordered': True}, CategoricalDtype(categories=['a', 'b', 'c'], ordered=True)), ({'type': 'string'}, 'object')])
def test_convert_json_field_to_pandas_type(self, inp, exp):
    field = {'name': 'foo'}
    field.update(inp)
    assert convert_json_field_to_pandas_type(field) == exp