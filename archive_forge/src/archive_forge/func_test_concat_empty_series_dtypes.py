import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('left,right,expected', [(np.bool_, np.int32, np.object_), (np.bool_, np.float32, np.object_), ('m8[ns]', np.bool_, np.object_), ('m8[ns]', np.int64, np.object_), ('M8[ns]', np.bool_, np.object_), ('M8[ns]', np.int64, np.object_), ('category', 'category', 'category'), ('category', 'object', 'object')])
def test_concat_empty_series_dtypes(self, left, right, expected):
    result = concat([Series(dtype=left), Series(dtype=right)])
    assert result.dtype == expected