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
@image_comparison(['connection_patch.png'], style='mpl20', remove_text=True)
def test_connection_patch():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    con = mpatches.ConnectionPatch(xyA=(0.1, 0.1), xyB=(0.9, 0.9), coordsA='data', coordsB='data', axesA=ax2, axesB=ax1, arrowstyle='->')
    ax2.add_artist(con)
    xyA = (0.6, 1.0)
    xyB = (0.0, 0.2)
    coordsA = 'axes fraction'
    coordsB = ax2.get_yaxis_transform()
    con = mpatches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA, coordsB=coordsB, arrowstyle='-')
    ax2.add_artist(con)