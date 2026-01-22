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
@mpl3d_image_comparison(['voxels-rgb-data.png'], style='mpl20')
def test_rgb_data(self):
    """Test with colors set to a 4d float array of rgb data."""
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x, y, z = np.indices((10, 10, 10))
    voxels = (x == y) | (y == z)
    colors = np.zeros((10, 10, 10, 3))
    colors[..., 0] = x / 9
    colors[..., 1] = y / 9
    colors[..., 2] = z / 9
    ax.voxels(voxels, facecolors=colors)