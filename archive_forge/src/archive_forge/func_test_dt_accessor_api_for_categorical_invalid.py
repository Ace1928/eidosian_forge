import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
def test_dt_accessor_api_for_categorical_invalid(self):
    invalid = Series([1, 2, 3]).astype('category')
    msg = 'Can only use .dt accessor with datetimelike'
    with pytest.raises(AttributeError, match=msg):
        invalid.dt
    assert not hasattr(invalid, 'str')