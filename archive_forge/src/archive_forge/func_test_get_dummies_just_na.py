import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_just_na(self, sparse):
    just_na_list = [np.nan]
    just_na_series = Series(just_na_list)
    just_na_series_index = Series(just_na_list, index=['A'])
    res_list = get_dummies(just_na_list, sparse=sparse)
    res_series = get_dummies(just_na_series, sparse=sparse)
    res_series_index = get_dummies(just_na_series_index, sparse=sparse)
    assert res_list.empty
    assert res_series.empty
    assert res_series_index.empty
    assert res_list.index.tolist() == [0]
    assert res_series.index.tolist() == [0]
    assert res_series_index.index.tolist() == ['A']