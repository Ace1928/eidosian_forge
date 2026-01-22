import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('colormap', ['jet', cm.jet])
def test_kde_colors_and_styles_subplots_cmap(self, colormap):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
    axes = df.plot(kind='kde', colormap=colormap, subplots=True)
    for ax, c in zip(axes, rgba_colors):
        _check_colors(ax.get_lines(), linecolors=[c])