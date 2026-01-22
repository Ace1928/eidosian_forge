import numpy as np
import pytest
from pandas import (
@pytest.fixture
def mixed_float_frame():
    """
    Fixture for DataFrame of different float types with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    df = DataFrame({col: np.random.default_rng(2).random(30, dtype=dtype) for col, dtype in zip(list('ABCD'), ['float32', 'float32', 'float32', 'float64'])}, index=Index([f'foo_{i}' for i in range(30)], dtype=object))
    df['C'] = df['C'].astype('float16')
    return df