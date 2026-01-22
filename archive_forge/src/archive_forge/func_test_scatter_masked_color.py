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
def test_scatter_masked_color():
    """
    Test color parameter usage with non-finite coordinate arrays.

    GH#26236
    """
    x = [np.nan, 1, 2, 1]
    y = [0, np.inf, 2, 1]
    z = [0, 1, -np.inf, 1]
    colors = [[0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1], [0.0, 0.0, 0.0, 1]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    path3d = ax.scatter(x, y, z, color=colors)
    assert len(path3d.get_offsets()) == len(super(type(path3d), path3d).get_facecolors())