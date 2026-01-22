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
@pytest.mark.parametrize('vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected', [('z', [[0.0, 1.142857, 0.0, -0.571429], [0.0, 0.0, 0.857143, -0.428571], [0.0, 0.0, 0.0, -10.0], [-1.142857, 0.0, 0.0, 10.571429]], [([0.05617978, 0.06329114], [-0.04213483, -0.04746835]), ([-0.06329114, 0.06329114], [-0.04746835, -0.04746835]), ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835])], [1, 0, 0]), ('y', [[1.142857, 0.0, 0.0, -0.571429], [0.0, 0.857143, 0.0, -0.428571], [0.0, 0.0, 0.0, -10.0], [0.0, 0.0, -1.142857, 10.571429]], [([-0.06329114, 0.06329114], [0.04746835, 0.04746835]), ([0.06329114, 0.06329114], [-0.04746835, 0.04746835]), ([-0.05617978, -0.06329114], [0.04213483, 0.04746835])], [2, 2, 0]), ('x', [[0.0, 0.0, 1.142857, -0.571429], [0.857143, 0.0, 0.0, -0.428571], [0.0, 0.0, 0.0, -10.0], [0.0, -1.142857, 0.0, 10.571429]], [([-0.06329114, -0.06329114], [0.04746835, -0.04746835]), ([0.06329114, 0.05617978], [0.04746835, 0.04213483]), ([0.06329114, -0.06329114], [0.04746835, 0.04746835])], [1, 2, 1])])
def test_view_init_vertical_axis(vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected):
    """
    Test the actual projection, axis lines and ticks matches expected values.

    Parameters
    ----------
    vertical_axis : str
        Axis to align vertically.
    proj_expected : ndarray
        Expected values from ax.get_proj().
    axis_lines_expected : tuple of arrays
        Edgepoints of the axis line. Expected values retrieved according
        to ``ax.get_[xyz]axis().line.get_data()``.
    tickdirs_expected : list of int
        indexes indicating which axis to create a tick line along.
    """
    rtol = 2e-06
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    ax.figure.canvas.draw()
    proj_actual = ax.get_proj()
    np.testing.assert_allclose(proj_expected, proj_actual, rtol=rtol)
    for i, axis in enumerate([ax.get_xaxis(), ax.get_yaxis(), ax.get_zaxis()]):
        axis_line_expected = axis_lines_expected[i]
        axis_line_actual = axis.line.get_data()
        np.testing.assert_allclose(axis_line_expected, axis_line_actual, rtol=rtol)
        tickdir_expected = tickdirs_expected[i]
        tickdir_actual = axis._get_tickdir('default')
        np.testing.assert_array_equal(tickdir_expected, tickdir_actual)