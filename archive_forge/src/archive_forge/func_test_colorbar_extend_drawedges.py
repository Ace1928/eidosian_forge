import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
@image_comparison(['extend_drawedges.png'], remove_text=True, style='mpl20')
def test_colorbar_extend_drawedges():
    params = [('both', 1, [[[1.1, 0], [1.1, 1]], [[2, 0], [2, 1]], [[2.9, 0], [2.9, 1]]]), ('min', 0, [[[1.1, 0], [1.1, 1]], [[2, 0], [2, 1]]]), ('max', 0, [[[2, 0], [2, 1]], [[2.9, 0], [2.9, 1]]]), ('neither', -1, [[[2, 0], [2, 1]]])]
    plt.rcParams['axes.linewidth'] = 2
    fig = plt.figure(figsize=(10, 4))
    subfigs = fig.subfigures(1, 2)
    for orientation, subfig in zip(['horizontal', 'vertical'], subfigs):
        if orientation == 'horizontal':
            axs = subfig.subplots(4, 1)
        else:
            axs = subfig.subplots(1, 4)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        for ax, (extend, coloroffset, res) in zip(axs, params):
            cmap = mpl.colormaps['viridis']
            bounds = np.arange(5)
            nb_colors = len(bounds) + coloroffset
            colors = cmap(np.linspace(100, 255, nb_colors).astype(int))
            cmap, norm = mcolors.from_levels_and_colors(bounds, colors, extend=extend)
            cbar = Colorbar(ax, cmap=cmap, norm=norm, orientation=orientation, drawedges=True)
            if orientation == 'horizontal':
                ax.set_xlim(1.1, 2.9)
            else:
                ax.set_ylim(1.1, 2.9)
                res = np.array(res)[:, :, [1, 0]]
            np.testing.assert_array_equal(cbar.dividers.get_segments(), res)