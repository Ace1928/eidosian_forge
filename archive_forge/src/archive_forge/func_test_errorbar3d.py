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
@mpl3d_image_comparison(['errorbar3d.png'], style='mpl20')
def test_errorbar3d():
    """Tests limits, color styling, and legend for 3D errorbars."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    d = [1, 2, 3, 4, 5]
    e = [0.5, 0.5, 0.5, 0.5, 0.5]
    ax.errorbar(x=d, y=d, z=d, xerr=e, yerr=e, zerr=e, capsize=3, zuplims=[False, True, False, True, True], zlolims=[True, False, False, True, False], yuplims=True, ecolor='purple', label='Error lines')
    ax.legend()