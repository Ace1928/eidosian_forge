from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_multifunc_select_col_integer_cols(self, df):
    df.columns = np.arange(len(df.columns))
    msg = 'Passing a dictionary to SeriesGroupBy.agg is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.groupby(1, as_index=False)[2].agg({'Q': np.mean})