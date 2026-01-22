import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_line_colors_and_styles_subplots_colormap_hex(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    custom_colors = ['#FF0000', '#0000FF', '#FFFF00', '#000000', '#FFFFFF']
    axes = df.plot(color=custom_colors, subplots=True)
    for ax, c in zip(axes, list(custom_colors)):
        _check_colors(ax.get_lines(), linecolors=[c])