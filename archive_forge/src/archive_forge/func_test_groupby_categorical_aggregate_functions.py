from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_categorical_aggregate_functions():
    dtype = pd.CategoricalDtype(categories=['small', 'big'], ordered=True)
    df = DataFrame([[1, 'small'], [1, 'big'], [2, 'small']], columns=['grp', 'description']).astype({'description': dtype})
    result = df.groupby('grp')['description'].max()
    expected = Series(['big', 'small'], index=Index([1, 2], name='grp'), name='description', dtype=pd.CategoricalDtype(categories=['small', 'big'], ordered=True))
    tm.assert_series_equal(result, expected)