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
@pytest.mark.parametrize('integer_data', [array([10], dtype='Int64'), Series(array([10], dtype='Int64'))])
def test_as_json_table_type_ext_integer_array_dtype(self, integer_data):
    assert as_json_table_type(integer_data.dtype) == 'integer'