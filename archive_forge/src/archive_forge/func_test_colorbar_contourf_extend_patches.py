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
@image_comparison(['contourf_extend_patches.png'], remove_text=True, style='mpl20')
def test_colorbar_contourf_extend_patches():
    params = [('both', 5, ['\\', '//']), ('min', 7, ['+']), ('max', 2, ['|', '-', '/', '\\', '//']), ('neither', 10, ['//', '\\', '||'])]
    plt.rcParams['axes.linewidth'] = 2
    fig = plt.figure(figsize=(10, 4))
    subfigs = fig.subfigures(1, 2)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    x = np.linspace(-2, 3, 50)
    y = np.linspace(-2, 3, 30)
    z = np.cos(x[np.newaxis, :]) + np.sin(y[:, np.newaxis])
    cmap = mpl.colormaps['viridis']
    for orientation, subfig in zip(['horizontal', 'vertical'], subfigs):
        axs = subfig.subplots(2, 2).ravel()
        for ax, (extend, levels, hatches) in zip(axs, params):
            cs = ax.contourf(x, y, z, levels, hatches=hatches, cmap=cmap, extend=extend)
            subfig.colorbar(cs, ax=ax, orientation=orientation, fraction=0.4, extendfrac=0.2, aspect=5)