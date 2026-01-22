import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('column, expected', [(None, ['width', 'length', 'height']), (['length', 'width', 'height'], ['length', 'width', 'height'])])
def test_hist_column_order_unchanged(self, column, expected):
    df = DataFrame({'width': [0.7, 0.2, 0.15, 0.2, 1.1], 'length': [1.5, 0.5, 1.2, 0.9, 3], 'height': [3, 0.5, 3.4, 2, 1]}, index=['pig', 'rabbit', 'duck', 'chicken', 'horse'])
    axes = _check_plot_works(df.hist, default_axes=True, column=column, layout=(1, 3))
    result = [axes[0, i].get_title() for i in range(3)]
    assert result == expected