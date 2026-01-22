import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_category_nansum(observed):
    categories = ['a', 'b', 'c']
    df = DataFrame({'A': pd.Categorical(['a', 'a', 'b'], categories=categories), 'B': [1, 2, 3]})
    msg = 'using SeriesGroupBy.sum'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.groupby('A', observed=observed).B.agg(np.nansum)
    expected = Series([3, 3, 0], index=pd.CategoricalIndex(['a', 'b', 'c'], categories=categories, name='A'), name='B')
    if observed:
        expected = expected[expected != 0]
    tm.assert_series_equal(result, expected)