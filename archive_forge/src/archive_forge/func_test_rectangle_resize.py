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
def test_rectangle_resize(ax):
    tool = widgets.RectangleSelector(ax, onselect=noop, interactive=True)
    click_and_drag(tool, start=(0, 10), end=(100, 120))
    assert tool.extents == (0.0, 100.0, 10.0, 120.0)
    extents = tool.extents
    xdata, ydata = (extents[1], extents[3])
    xdata_new, ydata_new = (xdata + 10, ydata + 5)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert tool.extents == (extents[0], xdata_new, extents[2], ydata_new)
    extents = tool.extents
    xdata, ydata = (extents[1], extents[2] + (extents[3] - extents[2]) / 2)
    xdata_new, ydata_new = (xdata + 10, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert tool.extents == (extents[0], xdata_new, extents[2], extents[3])
    extents = tool.extents
    xdata, ydata = (extents[0], extents[2] + (extents[3] - extents[2]) / 2)
    xdata_new, ydata_new = (xdata + 15, ydata)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert tool.extents == (xdata_new, extents[1], extents[2], extents[3])
    extents = tool.extents
    xdata, ydata = (extents[0], extents[2])
    xdata_new, ydata_new = (xdata + 20, ydata + 25)
    click_and_drag(tool, start=(xdata, ydata), end=(xdata_new, ydata_new))
    assert tool.extents == (xdata_new, extents[1], ydata_new, extents[3])