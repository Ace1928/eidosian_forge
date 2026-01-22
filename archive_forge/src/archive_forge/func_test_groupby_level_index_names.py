from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_level_index_names(self, axis):
    df = DataFrame({'exp': ['A'] * 3 + ['B'] * 3, 'var1': range(6)}).set_index('exp')
    if axis in (1, 'columns'):
        df = df.T
        depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    else:
        depr_msg = "The 'axis' keyword in DataFrame.groupby is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        df.groupby(level='exp', axis=axis)
    msg = f'level name foo is not the name of the {df._get_axis_name(axis)}'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df.groupby(level='foo', axis=axis)