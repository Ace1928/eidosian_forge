import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('colormap', ['k', 'red'])
def test_kde_colors_and_styles_subplots_single_col_str(self, colormap):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.plot(kind='kde', color=colormap, subplots=True)
    for ax in axes:
        _check_colors(ax.get_lines(), linecolors=[colormap])