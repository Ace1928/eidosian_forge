import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_multiple_axes_on_fig(self, hist_df):
    df = hist_df
    fig, axes = mpl.pyplot.subplots(2, 3)
    with tm.assert_produces_warning(UserWarning):
        returned = df.boxplot(column=['height', 'weight', 'category'], by='gender', return_type='axes', ax=axes[0])
    returned = np.array(list(returned.values))
    _check_axes_shape(returned, axes_num=3, layout=(1, 3))
    tm.assert_numpy_array_equal(returned, axes[0])
    assert returned[0].figure is fig
    with tm.assert_produces_warning(UserWarning):
        returned = df.groupby('classroom').boxplot(column=['height', 'weight', 'category'], return_type='axes', ax=axes[1])
    returned = np.array(list(returned.values))
    _check_axes_shape(returned, axes_num=3, layout=(1, 3))
    tm.assert_numpy_array_equal(returned, axes[1])
    assert returned[0].figure is fig