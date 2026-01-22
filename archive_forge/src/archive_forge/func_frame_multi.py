import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_multi():
    return DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=MultiIndex.from_product([[1, 2], [3, 4]]), columns=MultiIndex.from_product([[5, 6], [7, 8]]))