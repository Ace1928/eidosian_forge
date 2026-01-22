import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_labels():
    return DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=list('abcd'), columns=list('ABCD'))