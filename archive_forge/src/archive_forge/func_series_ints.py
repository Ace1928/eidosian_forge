import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_ints():
    return Series(np.random.default_rng(2).random(4), index=np.arange(0, 8, 2))