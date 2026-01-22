from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def frame_of_index_cols():
    """
    Fixture for DataFrame of columns that can be used for indexing

    Columns are ['A', 'B', 'C', 'D', 'E', ('tuple', 'as', 'label')];
    'A' & 'B' contain duplicates (but are jointly unique), the rest are unique.

         A      B  C         D         E  (tuple, as, label)
    0  foo    one  a  0.608477 -0.012500           -1.664297
    1  foo    two  b -0.633460  0.249614           -0.364411
    2  foo  three  c  0.615256  2.154968           -0.834666
    3  bar    one  d  0.234246  1.085675            0.718445
    4  bar    two  e  0.533841 -0.005702           -3.533912
    """
    df = DataFrame({'A': ['foo', 'foo', 'foo', 'bar', 'bar'], 'B': ['one', 'two', 'three', 'one', 'two'], 'C': ['a', 'b', 'c', 'd', 'e'], 'D': np.random.default_rng(2).standard_normal(5), 'E': np.random.default_rng(2).standard_normal(5), ('tuple', 'as', 'label'): np.random.default_rng(2).standard_normal(5)})
    return df