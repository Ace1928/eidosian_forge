import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_bounds():
    arr = np.random.default_rng(2).standard_normal(1000)
    factor = qcut(arr, 10, labels=False)
    assert len(np.unique(factor)) == 10