import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('by, layout, axes_num, res_layout', [['gender', (2, 1), 2, (2, 1)], ['gender', (3, -1), 2, (3, 1)], ['category', (4, 1), 4, (4, 1)], ['category', (2, -1), 4, (2, 2)], ['category', (3, -1), 4, (3, 2)], ['category', (-1, 4), 4, (1, 4)], ['classroom', (2, 2), 3, (2, 2)]])
def test_hist_layout_with_by(self, hist_df, by, layout, axes_num, res_layout):
    df = hist_df
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(df.height.hist, by=getattr(df, by), layout=layout)
    _check_axes_shape(axes, axes_num=axes_num, layout=res_layout)