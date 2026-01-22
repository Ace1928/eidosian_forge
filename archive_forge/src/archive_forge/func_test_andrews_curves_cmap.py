import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('df', ['iris', DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': np.random.default_rng(2).standard_normal(10), 'C': np.random.default_rng(2).standard_normal(10), 'Name': ['A'] * 10})])
def test_andrews_curves_cmap(self, request, df):
    from pandas.plotting import andrews_curves
    if isinstance(df, str):
        df = request.getfixturevalue(df)
    cmaps = [cm.jet(n) for n in np.linspace(0, 1, df['Name'].nunique())]
    ax = _check_plot_works(andrews_curves, frame=df, class_column='Name', color=cmaps)
    _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df['Name'][:10])