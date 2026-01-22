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
@pytest.mark.parametrize('decimal_data', [DecimalArray([decimal.Decimal(10)]), Series(DecimalArray([decimal.Decimal(10)]))])
def test_as_json_table_type_ext_decimal_array_dtype(self, decimal_data):
    assert as_json_table_type(decimal_data.dtype) == 'number'