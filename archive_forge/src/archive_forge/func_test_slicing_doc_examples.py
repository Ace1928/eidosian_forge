import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slicing_doc_examples(self):
    cats = Categorical(['a', 'b', 'b', 'b', 'c', 'c', 'c'], categories=['a', 'b', 'c'])
    idx = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values = [1, 2, 2, 2, 3, 4, 5]
    df = DataFrame({'cats': cats, 'values': values}, index=idx)
    result = df.iloc[2:4, :]
    expected = DataFrame({'cats': Categorical(['b', 'b'], categories=['a', 'b', 'c']), 'values': [2, 2]}, index=['j', 'k'])
    tm.assert_frame_equal(result, expected)
    result = df.iloc[2:4, :].dtypes
    expected = Series(['category', 'int64'], ['cats', 'values'], dtype=object)
    tm.assert_series_equal(result, expected)
    result = df.loc['h':'j', 'cats']
    expected = Series(Categorical(['a', 'b', 'b'], categories=['a', 'b', 'c']), index=['h', 'i', 'j'], name='cats')
    tm.assert_series_equal(result, expected)
    result = df.loc['h':'j', df.columns[0:1]]
    expected = DataFrame({'cats': Categorical(['a', 'b', 'b'], categories=['a', 'b', 'c'])}, index=['h', 'i', 'j'])
    tm.assert_frame_equal(result, expected)