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
@mpl3d_image_comparison(['voxels-edge-style.png'], style='mpl20')
def test_edge_style(self):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x, y, z = np.indices((5, 5, 4))
    voxels = (x - 2) ** 2 + (y - 2) ** 2 + (z - 1.5) ** 2 < 2.2 ** 2
    v = ax.voxels(voxels, linewidths=3, edgecolor='C1')
    v[max(v.keys())].set_edgecolor('C2')