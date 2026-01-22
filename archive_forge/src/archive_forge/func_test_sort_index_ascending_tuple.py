import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ascending', [(True, False), [True, False]])
def test_sort_index_ascending_tuple(self, ascending):
    df = DataFrame({'legs': [4, 2, 4, 2, 2]}, index=MultiIndex.from_tuples([('mammal', 'dog'), ('bird', 'duck'), ('mammal', 'horse'), ('bird', 'penguin'), ('mammal', 'kangaroo')], names=['class', 'animal']))
    result = df.sort_index(level=(0, 1), ascending=ascending)
    expected = DataFrame({'legs': [2, 2, 2, 4, 4]}, index=MultiIndex.from_tuples([('bird', 'penguin'), ('bird', 'duck'), ('mammal', 'kangaroo'), ('mammal', 'horse'), ('mammal', 'dog')], names=['class', 'animal']))
    tm.assert_frame_equal(result, expected)