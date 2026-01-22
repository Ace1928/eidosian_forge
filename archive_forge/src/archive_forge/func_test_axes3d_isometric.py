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
@mpl3d_image_comparison(['axes3d_isometric.png'], style='mpl20')
def test_axes3d_isometric():
    from itertools import combinations, product
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho', box_aspect=(4, 4, 4)))
    r = (-1, 1)
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if abs(s - e).sum() == r[1] - r[0]:
            ax.plot3D(*zip(s, e), c='k')
    ax.view_init(elev=np.degrees(np.arctan(1.0 / np.sqrt(2))), azim=-45, roll=0)
    ax.grid(True)