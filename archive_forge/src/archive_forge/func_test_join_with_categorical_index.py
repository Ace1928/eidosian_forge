import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_with_categorical_index(self):
    ix = ['a', 'b']
    id1 = pd.CategoricalIndex(ix, categories=ix)
    id2 = pd.CategoricalIndex(reversed(ix), categories=reversed(ix))
    df1 = DataFrame({'c1': ix}, index=id1)
    df2 = DataFrame({'c2': reversed(ix)}, index=id2)
    result = df1.join(df2)
    expected = DataFrame({'c1': ['a', 'b'], 'c2': ['a', 'b']}, index=pd.CategoricalIndex(['a', 'b'], categories=['a', 'b']))
    tm.assert_frame_equal(result, expected)