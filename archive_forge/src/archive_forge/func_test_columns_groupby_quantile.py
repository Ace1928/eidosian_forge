import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_columns_groupby_quantile():
    df = DataFrame(np.arange(12).reshape(3, -1), index=list('XYZ'), columns=pd.Series(list('ABAB'), name='col'))
    result = df.groupby('col', axis=1).quantile(q=[0.8, 0.2])
    expected = DataFrame([[1.6, 0.4, 2.6, 1.4], [5.6, 4.4, 6.6, 5.4], [9.6, 8.4, 10.6, 9.4]], index=list('XYZ'), columns=pd.MultiIndex.from_tuples([('A', 0.8), ('A', 0.2), ('B', 0.8), ('B', 0.2)], names=['col', None]))
    tm.assert_frame_equal(result, expected)