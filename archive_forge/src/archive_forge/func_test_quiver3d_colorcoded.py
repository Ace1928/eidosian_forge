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
@mpl3d_image_comparison(['quiver3d_colorcoded.png'], style='mpl20')
def test_quiver3d_colorcoded():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = y = dx = dz = np.zeros(10)
    z = dy = np.arange(10.0)
    color = plt.cm.Reds(dy / dy.max())
    ax.quiver(x, y, z, dx, dy, dz, colors=color)
    ax.set_ylim(0, 10)