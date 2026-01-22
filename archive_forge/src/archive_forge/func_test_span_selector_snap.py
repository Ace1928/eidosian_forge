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
def test_span_selector_snap(ax):

    def onselect(vmin, vmax):
        ax._got_onselect = True
    snap_values = np.arange(50) * 4
    tool = widgets.SpanSelector(ax, onselect, direction='horizontal', snap_values=snap_values)
    tool.extents = (17, 35)
    assert tool.extents == (16, 36)
    tool.snap_values = None
    assert tool.snap_values is None
    tool.extents = (17, 35)
    assert tool.extents == (17, 35)