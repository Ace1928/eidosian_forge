import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('kwargs', [{}, {'column': 'height', 'layout': (2, 2)}])
def test_grouped_hist_layout_by_warning(self, hist_df, kwargs):
    df = hist_df
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(df.hist, by='classroom', **kwargs)
    _check_axes_shape(axes, axes_num=3, layout=(2, 2))