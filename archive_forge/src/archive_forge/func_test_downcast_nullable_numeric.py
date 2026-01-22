import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, input_dtype, downcast, expected_dtype', (([1, 1], 'Int64', 'integer', 'Int8'), ([1.0, pd.NA], 'Float64', 'integer', 'Int8'), ([1.0, 1.1], 'Float64', 'integer', 'Float64'), ([1, pd.NA], 'Int64', 'integer', 'Int8'), ([450, 300], 'Int64', 'integer', 'Int16'), ([1, 1], 'Float64', 'integer', 'Int8'), ([np.iinfo(np.int64).max - 1, 1], 'Int64', 'integer', 'Int64'), ([1, 1], 'Int64', 'signed', 'Int8'), ([1.0, 1.0], 'Float32', 'signed', 'Int8'), ([1.0, 1.1], 'Float64', 'signed', 'Float64'), ([1, pd.NA], 'Int64', 'signed', 'Int8'), ([450, -300], 'Int64', 'signed', 'Int16'), ([np.iinfo(np.uint64).max - 1, 1], 'UInt64', 'signed', 'UInt64'), ([1, 1], 'Int64', 'unsigned', 'UInt8'), ([1.0, 1.0], 'Float32', 'unsigned', 'UInt8'), ([1.0, 1.1], 'Float64', 'unsigned', 'Float64'), ([1, pd.NA], 'Int64', 'unsigned', 'UInt8'), ([450, -300], 'Int64', 'unsigned', 'Int64'), ([-1, -1], 'Int32', 'unsigned', 'Int32'), ([1, 1], 'Float64', 'float', 'Float32'), ([1, 1.1], 'Float64', 'float', 'Float32'), ([1, 1], 'Float32', 'float', 'Float32'), ([1, 1.1], 'Float32', 'float', 'Float32')))
def test_downcast_nullable_numeric(data, input_dtype, downcast, expected_dtype):
    arr = pd.array(data, dtype=input_dtype)
    result = to_numeric(arr, downcast=downcast)
    expected = pd.array(data, dtype=expected_dtype)
    tm.assert_extension_array_equal(result, expected)