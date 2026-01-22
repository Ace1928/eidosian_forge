from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def df_strings():
    return pd.DataFrame({'a': np.random.default_rng(2).permutation(10), 'b': list(ascii_lowercase[:10]), 'c': np.random.default_rng(2).permutation(10).astype('float64')})