from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_quadmesh_cursor_data_multiple_points():
    x = [1, 2, 1, 2]
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(x, x, np.ones((3, 3)))
    fig.draw_without_rendering()
    xdata, ydata = (1.5, 1.5)
    x, y = mesh.get_transform().transform((xdata, ydata))
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)
    assert_array_equal(mesh.get_cursor_data(mouse_event), np.ones(9))