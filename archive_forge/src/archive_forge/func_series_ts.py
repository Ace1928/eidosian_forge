import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_ts():
    return Series(np.random.default_rng(2).standard_normal(4), index=date_range('20130101', periods=4))