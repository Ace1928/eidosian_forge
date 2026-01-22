import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def simple_multiindex_dataframe():
    """
    Factory function to create simple 3 x 3 dataframe with
    both columns and row MultiIndex using supplied data or
    random data by default.
    """
    data = np.random.default_rng(2).standard_normal((3, 3))
    return DataFrame(data, columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])