import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_grouped_plot_fignums_excluded_col(self):
    n = 10
    weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
    height = Series(np.random.default_rng(2).normal(60, 10, size=n))
    gender = np.random.default_rng(2).choice(['male', 'female'], size=n)
    df = DataFrame({'height': height, 'weight': weight, 'gender': gender})
    df.groupby('gender').hist()