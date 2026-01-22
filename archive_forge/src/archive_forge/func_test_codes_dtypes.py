import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_codes_dtypes(self):
    result = Categorical(['foo', 'bar', 'baz'])
    assert result.codes.dtype == 'int8'
    result = Categorical([f'foo{i:05d}' for i in range(400)])
    assert result.codes.dtype == 'int16'
    result = Categorical([f'foo{i:05d}' for i in range(40000)])
    assert result.codes.dtype == 'int32'
    result = Categorical(['foo', 'bar', 'baz'])
    assert result.codes.dtype == 'int8'
    result = result.add_categories([f'foo{i:05d}' for i in range(400)])
    assert result.codes.dtype == 'int16'
    result = result.remove_categories([f'foo{i:05d}' for i in range(300)])
    assert result.codes.dtype == 'int8'