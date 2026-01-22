import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_floats():
    return Series(np.random.default_rng(2).random(4), index=Index(range(0, 8, 2), dtype=np.float64))