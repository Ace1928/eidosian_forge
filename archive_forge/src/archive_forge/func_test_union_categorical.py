import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a, b, combined', [(list('abc'), list('abd'), list('abcabd')), ([0, 1, 2], [2, 3, 4], [0, 1, 2, 2, 3, 4]), ([0, 1.2, 2], [2, 3.4, 4], [0, 1.2, 2, 2, 3.4, 4]), (['b', 'b', np.nan, 'a'], ['a', np.nan, 'c'], ['b', 'b', np.nan, 'a', 'a', np.nan, 'c']), (pd.date_range('2014-01-01', '2014-01-05'), pd.date_range('2014-01-06', '2014-01-07'), pd.date_range('2014-01-01', '2014-01-07')), (pd.date_range('2014-01-01', '2014-01-05', tz='US/Central'), pd.date_range('2014-01-06', '2014-01-07', tz='US/Central'), pd.date_range('2014-01-01', '2014-01-07', tz='US/Central')), (pd.period_range('2014-01-01', '2014-01-05'), pd.period_range('2014-01-06', '2014-01-07'), pd.period_range('2014-01-01', '2014-01-07'))])
@pytest.mark.parametrize('box', [Categorical, CategoricalIndex, Series])
def test_union_categorical(self, a, b, combined, box):
    result = union_categoricals([box(Categorical(a)), box(Categorical(b))])
    expected = Categorical(combined)
    tm.assert_categorical_equal(result, expected)