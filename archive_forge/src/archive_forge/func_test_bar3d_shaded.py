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
@mpl3d_image_comparison(['bar3d_shaded.png'], style='mpl20')
def test_bar3d_shaded():
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = (x2d.ravel(), y2d.ravel())
    z = x2d + y2d + 1
    views = [(30, -60, 0), (30, 30, 30), (-30, 30, -90), (300, -30, 0)]
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(1, len(views), subplot_kw=dict(projection='3d'))
    for ax, (elev, azim, roll) in zip(axs, views):
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        ax.view_init(elev=elev, azim=azim, roll=roll)
    fig.canvas.draw()