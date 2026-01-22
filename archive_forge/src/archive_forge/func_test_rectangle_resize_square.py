import functools
import io
from unittest import mock
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
import matplotlib.colors as mcolors
import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing.widgets import (click_and_drag, do_event, get_ax,
import numpy as np
from numpy.testing import assert_allclose
import pytest
@pytest.mark.parametrize('add_state', [True, False])
def test_rectangle_resize_square(ax, add_state):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    click_and_drag(tool, start=(70, 65), end=(120, 115))
    assert tool.extents == (70.0, 120.0, 65.0, 115.0)
    if add_state:
        tool.add_state('square')
        use_key = None
    else:
        use_key = 'shift'
    extents = tool.extents
    xdata, ydata = (extents[1], extents[3])
    xdiff, ydiff = (10, 5)
    xdata_new, ydata_new = (xdata + xdiff, ydata + ydiff)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3] + xdiff)
    extents = tool.extents
    xdata, ydata = (extents[1], extents[2] + (extents[3] - extents[2]) / 2)
    xdiff = 10
    xdata_new, ydata_new = (xdata + xdiff, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3] + xdiff)
    extents = tool.extents
    xdata, ydata = (extents[1], extents[2] + (extents[3] - extents[2]) / 2)
    xdiff = -20
    xdata_new, ydata_new = (xdata + xdiff, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3] + xdiff)
    extents = tool.extents
    xdata, ydata = (extents[0], extents[2] + (extents[3] - extents[2]) / 2)
    xdiff = 15
    xdata_new, ydata_new = (xdata + xdiff, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (xdata_new, extents[1], extents[2], extents[3] - xdiff)
    extents = tool.extents
    xdata, ydata = (extents[0], extents[2] + (extents[3] - extents[2]) / 2)
    xdiff = -25
    xdata_new, ydata_new = (xdata + xdiff, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (xdata_new, extents[1], extents[2], extents[3] - xdiff)
    extents = tool.extents
    xdata, ydata = (extents[0], extents[2])
    xdiff, ydiff = (20, 25)
    xdata_new, ydata_new = (xdata + xdiff, ydata + ydiff)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new), key=use_key)
    assert tool.extents == (extents[0] + ydiff, extents[1], ydata_new, extents[3])