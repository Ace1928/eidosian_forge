from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_indicator_result_integrity(self, dfs_for_indicator):
    df1, df2 = dfs_for_indicator
    test2 = merge(df1, df2, on='col1', how='left', indicator=True)
    assert (test2._merge != 'right_only').all()
    test2 = df1.merge(df2, on='col1', how='left', indicator=True)
    assert (test2._merge != 'right_only').all()
    test3 = merge(df1, df2, on='col1', how='right', indicator=True)
    assert (test3._merge != 'left_only').all()
    test3 = df1.merge(df2, on='col1', how='right', indicator=True)
    assert (test3._merge != 'left_only').all()
    test4 = merge(df1, df2, on='col1', how='inner', indicator=True)
    assert (test4._merge == 'both').all()
    test4 = df1.merge(df2, on='col1', how='inner', indicator=True)
    assert (test4._merge == 'both').all()