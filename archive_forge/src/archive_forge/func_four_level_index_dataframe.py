import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.fixture
def four_level_index_dataframe():
    arr = np.array([[-0.5109, -2.3358, -0.4645, 0.05076, 0.364], [0.4473, 1.4152, 0.2834, 1.00661, 0.1744], [-0.6662, -0.5243, -0.358, 0.89145, 2.5838]])
    index = MultiIndex(levels=[['a', 'x'], ['b', 'q'], [10.0032, 20.0, 30.0], [3, 4, 5]], codes=[[0, 0, 1], [0, 1, 1], [0, 1, 2], [2, 1, 0]], names=['one', 'two', 'three', 'four'])
    return DataFrame(arr, index=index, columns=list('ABCDE'))