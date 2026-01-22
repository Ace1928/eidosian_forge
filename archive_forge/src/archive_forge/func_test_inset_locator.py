from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
@image_comparison(['inset_locator.png'], style='default', remove_text=True)
def test_inset_locator():
    fig, ax = plt.subplots(figsize=[5, 4])
    Z = cbook.get_sample_data('axes_grid/bivariate_normal.npy')
    extent = (-3, 4, -4, 3)
    Z2 = np.zeros((150, 150))
    ny, nx = Z.shape
    Z2[30:30 + ny, 30:30 + nx] = Z
    ax.imshow(Z2, extent=extent, interpolation='nearest', origin='lower')
    axins = zoomed_inset_axes(ax, zoom=6, loc='upper right')
    axins.imshow(Z2, extent=extent, interpolation='nearest', origin='lower')
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    x1, x2, y1, y2 = (-1.5, -0.9, -2.5, -1.9)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')
    asb = AnchoredSizeBar(ax.transData, 0.5, '0.5', loc='lower center', pad=0.1, borderpad=0.5, sep=5, frameon=False)
    ax.add_artist(asb)