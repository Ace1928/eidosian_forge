import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
def test_insert_invalid_loc(self, data):
    ub = len(data)
    with pytest.raises(IndexError):
        data.insert(ub + 1, data[0])
    with pytest.raises(IndexError):
        data.insert(-ub - 1, data[0])
    with pytest.raises(TypeError):
        data.insert(1.5, data[0])