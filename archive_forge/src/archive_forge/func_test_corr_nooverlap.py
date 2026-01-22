import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('meth', ['pearson', 'kendall', 'spearman'])
def test_corr_nooverlap(self, meth):
    pytest.importorskip('scipy')
    df = DataFrame({'A': [1, 1.5, 1, np.nan, np.nan, np.nan], 'B': [np.nan, np.nan, np.nan, 1, 1.5, 1], 'C': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    rs = df.corr(meth)
    assert isna(rs.loc['A', 'B'])
    assert isna(rs.loc['B', 'A'])
    assert rs.loc['A', 'A'] == 1
    assert rs.loc['B', 'B'] == 1
    assert isna(rs.loc['C', 'C'])