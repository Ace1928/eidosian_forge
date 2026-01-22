import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('column, expected_axes_num', [(None, 2), ('b', 1)])
def test_groupby_hist_frame_with_legend(self, column, expected_axes_num):
    expected_layout = (1, expected_axes_num)
    expected_labels = column or [['a'], ['b']]
    index = Index(15 * ['1'] + 15 * ['2'], name='c')
    df = DataFrame(np.random.default_rng(2).standard_normal((30, 2)), index=index, columns=['a', 'b'])
    g = df.groupby('c')
    for axes in g.hist(legend=True, column=column):
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        for ax, expected_label in zip(axes[0], expected_labels):
            _check_legend_labels(ax, expected_label)