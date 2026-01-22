import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_get_standard_colors_random_seed(self):
    df = DataFrame(np.zeros((10, 10)))
    plotting.parallel_coordinates(df, 0)
    rand1 = np.random.default_rng(None).random()
    plotting.parallel_coordinates(df, 0)
    rand2 = np.random.default_rng(None).random()
    assert rand1 != rand2