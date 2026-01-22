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
@mpl3d_image_comparison(['quiver3d.png'], style='mpl20')
def test_quiver3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    pivots = ['tip', 'middle', 'tail']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (pivot, color) in enumerate(zip(pivots, colors)):
        x, y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        u = -x
        v = -y
        w = -z
        z += 2 * i
        ax.quiver(x, y, z, u, v, w, length=1, pivot=pivot, color=color)
        ax.scatter(x, y, z, color=color)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 5)