import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.fixture
def hist_df():
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), columns=['A', 'B'])
    df['C'] = np.random.default_rng(2).choice(['a', 'b', 'c'], 30)
    df['D'] = np.random.default_rng(2).choice(['a', 'b', 'c'], 30)
    return df