from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('function', [lambda gr: gr.index, lambda gr: gr.index + 1 - 1])
def test_apply_function_index_return(function):
    df = DataFrame([1, 2, 2, 2, 1, 2, 3, 1, 3, 1], columns=['id'])
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('id').apply(function)
    expected = Series([Index([0, 4, 7, 9]), Index([1, 2, 3, 5]), Index([6, 8])], index=Index([1, 2, 3], name='id'))
    tm.assert_series_equal(result, expected)