from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('cat_columns', ['a', 'b', ['a', 'b']])
@pytest.mark.parametrize('keys', ['a', 'b', ['a', 'b']])
def test_groupby_default_depr(cat_columns, keys):
    df = DataFrame({'a': [1, 1, 2, 3], 'b': [4, 5, 6, 7]})
    df[cat_columns] = df[cat_columns].astype('category')
    msg = 'The default of observed=False is deprecated'
    klass = FutureWarning if set(cat_columns) & set(keys) else None
    with tm.assert_produces_warning(klass, match=msg):
        df.groupby(keys)