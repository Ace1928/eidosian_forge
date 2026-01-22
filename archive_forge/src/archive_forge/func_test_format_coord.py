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
def test_format_coord():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(10)
    ax.plot(x, np.sin(x))
    xv = 0.1
    yv = 0.1
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'
    ax.view_init(roll=30, vertical_axis='y')
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x pane=9.1875, y=0.9761, z=0.1291'
    ax.view_init()
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'
    ax.set_proj_type('ortho')
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=10.8869, y pane=1.0417, z=0.1528'
    ax.set_proj_type('persp', focal_length=0.1)
    fig.canvas.draw()
    assert ax.format_coord(xv, yv) == 'x=9.0620, y pane=1.0417, z=0.1110'