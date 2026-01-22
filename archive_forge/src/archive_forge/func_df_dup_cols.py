import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
@pytest.fixture
def df_dup_cols(self):
    dups = ['A', 'A', 'C', 'D']
    df = DataFrame(np.arange(12).reshape(3, 4), columns=dups, dtype='float64')
    return df