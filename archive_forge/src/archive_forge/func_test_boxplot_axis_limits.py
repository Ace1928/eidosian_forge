import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_axis_limits(self, hist_df):
    df = hist_df.copy()
    df['age'] = np.random.default_rng(2).integers(1, 20, df.shape[0])
    height_ax, weight_ax = df.boxplot(['height', 'weight'], by='category')
    _check_ax_limits(df['height'], height_ax)
    _check_ax_limits(df['weight'], weight_ax)
    assert weight_ax._sharey == height_ax