import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_layout(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(df.hist, grid=False)
    _check_axes_shape(axes, axes_num=3, layout=(2, 2))
    assert not axes[1, 1].get_visible()
    _check_plot_works(df[[2]].hist)