from collections import OrderedDict
import datetime as dt
import decimal
from io import StringIO
import json
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.string_ import StringDtype
from pandas.core.series import Series
from pandas.tests.extension.date import (
from pandas.tests.extension.decimal.array import (
from pandas.io.json._table_schema import (
def test_build_date_series(self, da):
    s = Series(da, name='a')
    s.index.name = 'id'
    result = s.to_json(orient='table', date_format='iso')
    result = json.loads(result, object_pairs_hook=OrderedDict)
    assert 'pandas_version' in result['schema']
    result['schema'].pop('pandas_version')
    fields = [{'name': 'id', 'type': 'integer'}, {'name': 'a', 'type': 'any', 'extDtype': 'DateDtype'}]
    schema = {'fields': fields, 'primaryKey': ['id']}
    expected = OrderedDict([('schema', schema), ('data', [OrderedDict([('id', 0), ('a', '2021-10-10T00:00:00.000')])])])
    assert result == expected