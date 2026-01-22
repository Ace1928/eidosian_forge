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
@mpl3d_image_comparison(['add_collection3d_zs_scalar.png'], style='mpl20')
def test_add_collection3d_zs_scalar():
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    norm = plt.Normalize(0, 2 * np.pi)
    lc = LineCollection(segments, cmap='twilight', norm=norm)
    lc.set_array(theta)
    line = ax.add_collection3d(lc, zs=z)
    assert line is not None
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(0, 2)