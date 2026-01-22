import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_radviz_color_cmap(self, iris):
    from matplotlib import cm
    from pandas.plotting import radviz
    df = iris
    ax = _check_plot_works(radviz, frame=df, class_column='Name', colormap=cm.jet)
    cmaps = [cm.jet(n) for n in np.linspace(0, 1, df['Name'].nunique())]
    patches = [p for p in ax.patches[:20] if p.get_label() != '']
    _check_colors(patches, facecolors=cmaps, mapping=df['Name'][:10])