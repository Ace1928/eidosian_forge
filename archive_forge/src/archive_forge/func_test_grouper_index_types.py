from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('index', [Index(list('abcde')), Index(np.arange(5)), Index(np.arange(5, dtype=float)), date_range('2020-01-01', periods=5), period_range('2020-01-01', periods=5)])
def test_grouper_index_types(self, index):
    df = DataFrame(np.arange(10).reshape(5, 2), columns=list('AB'), index=index)
    df.groupby(list('abcde'), group_keys=False).apply(lambda x: x)
    df.index = df.index[::-1]
    df.groupby(list('abcde'), group_keys=False).apply(lambda x: x)