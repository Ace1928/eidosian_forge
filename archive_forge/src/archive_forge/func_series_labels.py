import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_labels():
    return Series(np.random.default_rng(2).standard_normal(4), index=list('abcd'))