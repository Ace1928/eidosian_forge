import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
@pytest.fixture
def date_range_frame():
    """
    Fixture for DataFrame of ints with date_range index

    Columns are ['A', 'B'].
    """
    N = 50
    rng = date_range('1/1/1990', periods=N, freq='53s')
    return DataFrame({'A': np.arange(N), 'B': np.arange(N)}, index=rng)