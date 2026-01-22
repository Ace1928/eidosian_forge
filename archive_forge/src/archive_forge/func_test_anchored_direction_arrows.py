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
@image_comparison(['anchored_direction_arrows.png'], tol=0 if platform.machine() == 'x86_64' else 0.01, style=('classic', '_classic_test_patch'))
def test_anchored_direction_arrows():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), interpolation='nearest')
    simple_arrow = AnchoredDirectionArrows(ax.transAxes, 'X', 'Y')
    ax.add_artist(simple_arrow)