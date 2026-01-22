import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith(self):
    pytest.importorskip('scipy')
    index = ['a', 'b', 'c', 'd', 'e']
    columns = ['one', 'two', 'three', 'four']
    df1 = tm.SubclassedDataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=index, columns=columns)
    df2 = tm.SubclassedDataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index[:4], columns=columns)
    correls = df1.corrwith(df2, axis=1, drop=True, method='kendall')
    assert isinstance(correls, tm.SubclassedSeries)