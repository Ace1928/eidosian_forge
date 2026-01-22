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
@mpl3d_image_comparison(['surface3d_masked.png'], style='mpl20')
def test_surface3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8]
    x, y = np.meshgrid(x, y)
    matrix = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1], [-1, -1.0, 4, 5, 6, 8, 6, 5, 4, 3, -1.0], [-1, -1.0, 7, 8, 11, 12, 11, 8, 7, -1.0, -1.0], [-1, -1.0, 8, 9, 10, 16, 10, 9, 10, 7, -1.0], [-1, -1.0, -1.0, 12, 16, 20, 16, 12, 11, -1.0, -1.0], [-1, -1.0, -1.0, -1.0, 22, 24, 22, 20, 18, -1.0, -1.0], [-1, -1.0, -1.0, -1.0, -1.0, 28, 26, 25, -1.0, -1.0, -1.0]])
    z = np.ma.masked_less(matrix, 0)
    norm = mcolors.Normalize(vmax=z.max(), vmin=z.min())
    colors = mpl.colormaps['plasma'](norm(z))
    ax.plot_surface(x, y, z, facecolors=colors)
    ax.view_init(30, -80, 0)