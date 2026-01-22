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
def test_line3d_set_get_data_3d():
    x, y, z = ([0, 1], [2, 3], [4, 5])
    x2, y2, z2 = ([6, 7], [8, 9], [10, 11])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    lines = ax.plot(x, y, z)
    line = lines[0]
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_data_3d(x2, y2, z2)
    np.testing.assert_array_equal((x2, y2, z2), line.get_data_3d())
    line.set_xdata(x)
    line.set_ydata(y)
    line.set_3d_properties(zs=z, zdir='z')
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_3d_properties(zs=0, zdir='z')
    np.testing.assert_array_equal((x, y, np.zeros_like(z)), line.get_data_3d())