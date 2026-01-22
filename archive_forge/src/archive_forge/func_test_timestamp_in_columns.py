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
def test_timestamp_in_columns(self):
    df = DataFrame([[1, 2]], columns=[pd.Timestamp('2016'), pd.Timedelta(10, unit='s')])
    result = df.to_json(orient='table')
    js = json.loads(result)
    assert js['schema']['fields'][1]['name'] == '2016-01-01T00:00:00.000'
    assert js['schema']['fields'][2]['name'] == 'P0DT0H0M10S'