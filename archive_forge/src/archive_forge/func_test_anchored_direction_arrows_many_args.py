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
@image_comparison(['anchored_direction_arrows_many_args.png'], style=('classic', '_classic_test_patch'))
def test_anchored_direction_arrows_many_args():
    fig, ax = plt.subplots()
    ax.imshow(np.ones((10, 10)))
    direction_arrows = AnchoredDirectionArrows(ax.transAxes, 'A', 'B', loc='upper right', color='red', aspect_ratio=-0.5, pad=0.6, borderpad=2, frameon=True, alpha=0.7, sep_x=-0.06, sep_y=-0.08, back_length=0.1, head_width=9, head_length=10, tail_width=5)
    ax.add_artist(direction_arrows)