import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('cols', [2, -1])
def test_grouped_box_layout_works(self, hist_df, cols):
    df = hist_df
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        _check_plot_works(df.groupby('category').boxplot, column='height', layout=(3, cols), return_type='dict')
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(3, 2))