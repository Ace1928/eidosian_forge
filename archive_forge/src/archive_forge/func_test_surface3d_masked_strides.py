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
@mpl3d_image_comparison(['surface3d_masked_strides.png'], style='mpl20')
def test_surface3d_masked_strides():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = np.mgrid[-6:6.1:1, -6:6.1:1]
    z = np.ma.masked_less(x * y, 2)
    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    ax.view_init(60, -45, 0)