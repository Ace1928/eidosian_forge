import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_default_color_cycle(self):
    import cycler
    colors = list('rgbk')
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', colors)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
    ax = df.plot()
    expected = _unpack_cycler(plt.rcParams)[:3]
    _check_colors(ax.get_lines(), linecolors=expected)