import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_line_colors_and_styles_subplots(self):
    default_colors = _unpack_cycler(mpl.pyplot.rcParams)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.plot(subplots=True)
    for ax, c in zip(axes, list(default_colors)):
        _check_colors(ax.get_lines(), linecolors=[c])