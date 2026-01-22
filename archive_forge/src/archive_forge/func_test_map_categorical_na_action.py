from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_action, expected', ([None, Series(['A', 'B', 'nan'], name='XX')], ['ignore', Series(['A', 'B', np.nan], name='XX', dtype=pd.CategoricalDtype(list('DCBA'), True))]))
def test_map_categorical_na_action(na_action, expected):
    dtype = pd.CategoricalDtype(list('DCBA'), ordered=True)
    values = pd.Categorical(list('AB') + [np.nan], dtype=dtype)
    s = Series(values, name='XX')
    result = s.map(str, na_action=na_action)
    tm.assert_series_equal(result, expected)