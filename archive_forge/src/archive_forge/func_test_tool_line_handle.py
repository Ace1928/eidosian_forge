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
def test_tool_line_handle(ax):
    positions = [20, 30, 50]
    tool_line_handle = widgets.ToolLineHandles(ax, positions, 'horizontal', useblit=False)
    for artist in tool_line_handle.artists:
        assert not artist.get_animated()
        assert not artist.get_visible()
    tool_line_handle.set_visible(True)
    tool_line_handle.set_animated(True)
    for artist in tool_line_handle.artists:
        assert artist.get_animated()
        assert artist.get_visible()
    assert tool_line_handle.positions == positions