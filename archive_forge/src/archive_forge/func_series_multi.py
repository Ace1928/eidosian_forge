import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_multi():
    return Series(np.random.default_rng(2).random(4), index=MultiIndex.from_product([[1, 2], [3, 4]]))