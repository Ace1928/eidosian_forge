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
def test_vbox_divider():
    arr1 = np.arange(20).reshape((4, 5))
    arr2 = np.arange(20).reshape((5, 4))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(arr1)
    ax2.imshow(arr2)
    pad = 0.5
    divider = VBoxDivider(fig, 111, horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)], vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])
    ax1.set_axes_locator(divider.new_locator(0))
    ax2.set_axes_locator(divider.new_locator(2))
    fig.canvas.draw()
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    assert p1.width == p2.width
    assert p1.height / p2.height == pytest.approx((4 / 5) ** 2)