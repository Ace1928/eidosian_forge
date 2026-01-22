import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('colormap', ['jet', cm.jet])
def test_boxplot_colors_cmap(self, colormap):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    bp = df.plot.box(colormap=colormap, return_type='dict')
    jet_colors = [cm.jet(n) for n in np.linspace(0, 1, 3)]
    _check_colors_box(bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0])