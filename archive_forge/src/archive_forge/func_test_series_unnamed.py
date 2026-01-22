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
def test_series_unnamed(self):
    result = build_table_schema(pd.Series([1, 2, 3]), version=False)
    expected = {'fields': [{'name': 'index', 'type': 'integer'}, {'name': 'values', 'type': 'integer'}], 'primaryKey': ['index']}
    assert result == expected