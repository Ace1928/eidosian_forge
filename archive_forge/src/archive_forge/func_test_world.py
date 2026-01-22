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
def test_world():
    xmin, xmax = (100, 120)
    ymin, ymax = (-100, 100)
    zmin, zmax = (0.1, 0.2)
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    np.testing.assert_allclose(M, [[0.05, 0, 0, -5], [0, 0.005, 0, 0.5], [0, 0, 10.0, -1], [0, 0, 0, 1]])