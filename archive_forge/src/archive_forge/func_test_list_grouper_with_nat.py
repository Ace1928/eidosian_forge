from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_list_grouper_with_nat(self):
    df = DataFrame({'date': date_range('1/1/2011', periods=365, freq='D')})
    df.iloc[-1] = pd.NaT
    grouper = Grouper(key='date', freq='YS')
    result = df.groupby([grouper])
    expected = {Timestamp('2011-01-01'): Index(list(range(364)))}
    tm.assert_dict_equal(result.groups, expected)
    result = df.groupby(grouper)
    expected = {Timestamp('2011-01-01'): 365}
    tm.assert_dict_equal(result.groups, expected)