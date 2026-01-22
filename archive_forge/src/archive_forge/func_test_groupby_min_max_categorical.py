import builtins
import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('func', ['first', 'last', 'min', 'max'])
def test_groupby_min_max_categorical(func):
    df = DataFrame({'col1': pd.Categorical(['A'], categories=list('AB'), ordered=True), 'col2': pd.Categorical([1], categories=[1, 2], ordered=True), 'value': 0.1})
    result = getattr(df.groupby('col1', observed=False), func)()
    idx = pd.CategoricalIndex(data=['A', 'B'], name='col1', ordered=True)
    expected = DataFrame({'col2': pd.Categorical([1, None], categories=[1, 2], ordered=True), 'value': [0.1, None]}, index=idx)
    tm.assert_frame_equal(result, expected)