import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_ints():
    return DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=np.arange(0, 8, 2), columns=np.arange(0, 12, 3))