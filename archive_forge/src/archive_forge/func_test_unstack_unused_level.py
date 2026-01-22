from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.parametrize('cols', [['A', 'C'], slice(None)])
def test_unstack_unused_level(self, cols):
    df = DataFrame([[2010, 'a', 'I'], [2011, 'b', 'II']], columns=['A', 'B', 'C'])
    ind = df.set_index(['A', 'B', 'C'], drop=False)
    selection = ind.loc[(slice(None), slice(None), 'I'), cols]
    result = selection.unstack()
    expected = ind.iloc[[0]][cols]
    expected.columns = MultiIndex.from_product([expected.columns, ['I']], names=[None, 'C'])
    expected.index = expected.index.droplevel('C')
    tm.assert_frame_equal(result, expected)