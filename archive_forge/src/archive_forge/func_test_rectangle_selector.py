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
@pytest.mark.parametrize('kwargs', [dict(), dict(useblit=True, button=1), dict(minspanx=10, minspany=10, spancoords='pixels'), dict(props=dict(fill=True))])
def test_rectangle_selector(ax, kwargs):
    onselect = mock.Mock(spec=noop, return_value=None)
    tool = widgets.RectangleSelector(ax, onselect, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)
    do_event(tool, 'release', xdata=250, ydata=250, button=1)
    if kwargs.get('drawtype', None) not in ['line', 'none']:
        assert_allclose(tool.geometry, [[100.0, 100, 199, 199, 100], [100, 199, 199, 100, 100]], err_msg=tool.geometry)
    onselect.assert_called_once()
    (epress, erelease), kwargs = onselect.call_args
    assert epress.xdata == 100
    assert epress.ydata == 100
    assert erelease.xdata == 199
    assert erelease.ydata == 199
    assert kwargs == {}