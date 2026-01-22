import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('linecolors', [('#556270', '#4ECDC4', '#C7F464'), ['dodgerblue', 'aquamarine', 'seagreen']])
@pytest.mark.parametrize('df', ['iris', DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': np.random.default_rng(2).standard_normal(10), 'C': np.random.default_rng(2).standard_normal(10), 'Name': ['A'] * 10})])
def test_andrews_curves_linecolors(self, request, df, linecolors):
    from pandas.plotting import andrews_curves
    if isinstance(df, str):
        df = request.getfixturevalue(df)
    ax = _check_plot_works(andrews_curves, frame=df, class_column='Name', color=linecolors)
    _check_colors(ax.get_lines()[:10], linecolors=linecolors, mapping=df['Name'][:10])