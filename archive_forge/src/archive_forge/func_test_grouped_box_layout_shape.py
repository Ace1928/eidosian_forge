import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_layout_shape(self, hist_df):
    df = hist_df
    df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], return_type='dict')
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))