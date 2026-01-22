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
@check_figures_equal(extensions=['png'])
def test_patch_collection_modification(fig_test, fig_ref):
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3)
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.add_collection3d(c)
    c.set_edgecolor('C2')
    c.set_facecolor(facecolors)
    c.set_alpha(0.7)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0.0, 0.5, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3, edgecolor='C2', facecolor=facecolors, alpha=0.7, depthshade=False)
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.add_collection3d(c)