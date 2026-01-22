import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('top', [True, False])
@pytest.mark.parametrize('bottom', [True, False])
@pytest.mark.parametrize('left', [True, False])
@pytest.mark.parametrize('right', [True, False])
def test_subplots_hide_ticklabels(top, bottom, left, right):
    with plt.rc_context({'xtick.labeltop': top, 'xtick.labelbottom': bottom, 'ytick.labelleft': left, 'ytick.labelright': right}):
        axs = plt.figure().subplots(3, 3, sharex=True, sharey=True)
    for (i, j), ax in np.ndenumerate(axs):
        xtop = ax.xaxis._major_tick_kw['label2On']
        xbottom = ax.xaxis._major_tick_kw['label1On']
        yleft = ax.yaxis._major_tick_kw['label1On']
        yright = ax.yaxis._major_tick_kw['label2On']
        assert xtop == (top and i == 0)
        assert xbottom == (bottom and i == 2)
        assert yleft == (left and j == 0)
        assert yright == (right and j == 2)