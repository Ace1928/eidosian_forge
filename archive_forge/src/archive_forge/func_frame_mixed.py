import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_mixed():
    return DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=[2, 4, 'null', 8])