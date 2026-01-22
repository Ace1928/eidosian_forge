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
def test_quadmesh_cursor_data():
    x = np.arange(4)
    X = x[:, None] * x[None, :]
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X)
    mesh._A = None
    fig.draw_without_rendering()
    xdata, ydata = (0.5, 0.5)
    x, y = mesh.get_transform().transform((xdata, ydata))
    mouse_event = SimpleNamespace(xdata=xdata, ydata=ydata, x=x, y=y)
    assert mesh.get_cursor_data(mouse_event) is None
    mesh.set_array(np.ones(X.shape))
    assert_array_equal(mesh.get_cursor_data(mouse_event), [1])