import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('nullable_column', [pd.array([1, 2, 3]), pd.array([1, 2, None])])
@pytest.mark.parametrize('other_column', [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])])
@pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
def test_corr_nullable_integer(self, nullable_column, other_column, method):
    pytest.importorskip('scipy')
    data = DataFrame({'a': nullable_column, 'b': other_column})
    result = data.corr(method=method)
    expected = DataFrame(np.ones((2, 2)), columns=['a', 'b'], index=['a', 'b'])
    tm.assert_frame_equal(result, expected)