import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('plot_method', ['scatter', 'hexbin'])
def test_pass_colormap_instance(df, plot_method):
    cmap = mpl.colors.ListedColormap([[1, 1, 1], [0, 0, 0]])
    df['c'] = df.A + df.B
    kwargs = {'x': 'A', 'y': 'B', 'c': 'c', 'colormap': cmap}
    if plot_method == 'hexbin':
        kwargs['C'] = kwargs.pop('c')
    getattr(df.plot, plot_method)(**kwargs)