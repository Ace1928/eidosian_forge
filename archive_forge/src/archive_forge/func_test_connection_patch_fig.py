import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@check_figures_equal(extensions=['png'])
def test_connection_patch_fig(fig_test, fig_ref):
    ax1, ax2 = fig_test.subplots(1, 2)
    con = mpatches.ConnectionPatch(xyA=(0.3, 0.2), coordsA='data', axesA=ax1, xyB=(-30, -20), coordsB='figure pixels', arrowstyle='->', shrinkB=5)
    fig_test.add_artist(con)
    ax1, ax2 = fig_ref.subplots(1, 2)
    bb = fig_ref.bbox
    plt.rcParams['savefig.dpi'] = plt.rcParams['figure.dpi']
    con = mpatches.ConnectionPatch(xyA=(0.3, 0.2), coordsA='data', axesA=ax1, xyB=(bb.width - 30, bb.height - 20), coordsB='figure pixels', arrowstyle='->', shrinkB=5)
    fig_ref.add_artist(con)