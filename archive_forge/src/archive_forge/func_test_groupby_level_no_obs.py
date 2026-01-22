import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_level_no_obs(self):
    midx = MultiIndex.from_tuples([('f1', 's1'), ('f1', 's2'), ('f2', 's1'), ('f2', 's2'), ('f3', 's1'), ('f3', 's2')])
    df = DataFrame([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], columns=midx)
    df1 = df.loc(axis=1)[df.columns.map(lambda u: u[0] in ['f2', 'f3'])]
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        grouped = df1.groupby(axis=1, level=0)
    result = grouped.sum()
    assert (result.columns == ['f2', 'f3']).all()