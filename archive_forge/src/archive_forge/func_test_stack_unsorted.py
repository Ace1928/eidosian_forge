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
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_unsorted(self, future_stack):
    PAE = ['ITA', 'FRA']
    VAR = ['A1', 'A2']
    TYP = ['CRT', 'DBT', 'NET']
    MI = MultiIndex.from_product([PAE, VAR, TYP], names=['PAE', 'VAR', 'TYP'])
    V = list(range(len(MI)))
    DF = DataFrame(data=V, index=MI, columns=['VALUE'])
    DF = DF.unstack(['VAR', 'TYP'])
    DF.columns = DF.columns.droplevel(0)
    DF.loc[:, ('A0', 'NET')] = 9999
    result = DF.stack(['VAR', 'TYP'], future_stack=future_stack).sort_index()
    expected = DF.sort_index(axis=1).stack(['VAR', 'TYP'], future_stack=future_stack).sort_index()
    tm.assert_series_equal(result, expected)