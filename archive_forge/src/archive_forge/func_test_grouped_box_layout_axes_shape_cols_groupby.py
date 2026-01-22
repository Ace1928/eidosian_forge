import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('cols, res', [[4, 4], [-1, 3]])
def test_grouped_box_layout_axes_shape_cols_groupby(self, hist_df, cols, res):
    df = hist_df
    df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], layout=(1, cols), return_type='dict')
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(1, res))