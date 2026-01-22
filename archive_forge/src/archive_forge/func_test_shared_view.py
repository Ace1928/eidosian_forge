import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@check_figures_equal(extensions=['png'])
def test_shared_view(fig_test, fig_ref):
    elev, azim, roll = (5, 20, 30)
    ax1 = fig_test.add_subplot(131, projection='3d')
    ax2 = fig_test.add_subplot(132, projection='3d', shareview=ax1)
    ax3 = fig_test.add_subplot(133, projection='3d')
    ax3.shareview(ax1)
    ax2.view_init(elev=elev, azim=azim, roll=roll, share=True)
    for subplot_num in (131, 132, 133):
        ax = fig_ref.add_subplot(subplot_num, projection='3d')
        ax.view_init(elev=elev, azim=azim, roll=roll)