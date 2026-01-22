import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_first_multi_key_groupby_categorical():
    df = DataFrame({'A': [1, 1, 1, 2, 2], 'B': [100, 100, 200, 100, 100], 'C': ['apple', 'orange', 'mango', 'mango', 'orange'], 'D': ['jupiter', 'mercury', 'mars', 'venus', 'venus']})
    df = df.astype({'D': 'category'})
    result = df.groupby(by=['A', 'B']).first()
    expected = DataFrame({'C': ['apple', 'mango', 'mango'], 'D': Series(['jupiter', 'mars', 'venus']).astype(pd.CategoricalDtype(['jupiter', 'mars', 'mercury', 'venus']))})
    expected.index = MultiIndex.from_tuples([(1, 100), (1, 200), (2, 100)], names=['A', 'B'])
    tm.assert_frame_equal(result, expected)