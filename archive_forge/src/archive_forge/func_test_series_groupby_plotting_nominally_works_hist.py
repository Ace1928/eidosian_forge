import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_series_groupby_plotting_nominally_works_hist(self):
    n = 10
    height = Series(np.random.default_rng(2).normal(60, 10, size=n))
    gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
    height.groupby(gender).hist()