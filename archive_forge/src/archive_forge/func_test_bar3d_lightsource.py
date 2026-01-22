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
def test_bar3d_lightsource():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ls = mcolors.LightSource(azdeg=0, altdeg=90)
    length, width = (3, 4)
    area = length * width
    x, y = np.meshgrid(np.arange(length), np.arange(width))
    x = x.ravel()
    y = y.ravel()
    dz = x + y
    color = [cm.coolwarm(i / area) for i in range(area)]
    collection = ax.bar3d(x=x, y=y, z=0, dx=1, dy=1, dz=dz, color=color, shade=True, lightsource=ls)
    np.testing.assert_array_max_ulp(color, collection._facecolor3d[1::6], 4)