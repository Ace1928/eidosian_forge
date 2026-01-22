import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('operation, expected', [('min', 'a'), ('max', 'b')])
def test_reductions_series_strings(operation, expected):
    ser = Series(['a', 'b'], dtype='string')
    res_operation_serie = getattr(ser, operation)()
    assert res_operation_serie == expected