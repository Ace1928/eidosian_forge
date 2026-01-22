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
@mpl3d_image_comparison(['voxels-alpha.png'], style='mpl20')
def test_alpha(self):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x, y, z = np.indices((10, 10, 10))
    v1 = x == y
    v2 = np.abs(x - y) < 2
    voxels = v1 | v2
    colors = np.zeros((10, 10, 10, 4))
    colors[v2] = [1, 0, 0, 0.5]
    colors[v1] = [0, 1, 0, 0.5]
    v = ax.voxels(voxels, facecolors=colors)
    assert type(v) is dict
    for coord, poly in v.items():
        assert voxels[coord], 'faces returned for absent voxel'
        assert isinstance(poly, art3d.Poly3DCollection)