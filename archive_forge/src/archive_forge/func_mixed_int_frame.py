import numpy as np
import pytest
from pandas import (
@pytest.fixture
def mixed_int_frame():
    """
    Fixture for DataFrame of different int types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame({col: np.ones(30, dtype=dtype) for col, dtype in zip(list('ABCD'), ['int32', 'uint64', 'uint8', 'int64'])}, index=Index([f'foo_{i}' for i in range(30)], dtype=object))