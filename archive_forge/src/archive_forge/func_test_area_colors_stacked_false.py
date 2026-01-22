import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_area_colors_stacked_false(self):
    from matplotlib import cm
    from matplotlib.collections import PolyCollection
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    jet_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
    ax = df.plot.area(colormap=cm.jet, stacked=False)
    _check_colors(ax.get_lines(), linecolors=jet_colors)
    poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
    jet_with_alpha = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
    _check_colors(poly, facecolors=jet_with_alpha)
    handles, _ = ax.get_legend_handles_labels()
    linecolors = jet_with_alpha
    _check_colors(handles[:len(jet_colors)], linecolors=linecolors)
    for h in handles:
        assert h.get_alpha() == 0.5