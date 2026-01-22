import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_itertuples_index_false(self):
    df = DataFrame({'floats': np.random.default_rng(2).standard_normal(5), 'ints': range(5)}, columns=['floats', 'ints'])
    for tup in df.itertuples(index=False):
        assert isinstance(tup[1], int)