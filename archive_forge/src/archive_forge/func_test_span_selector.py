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
@pytest.mark.parametrize('orientation, onmove_callback, kwargs', [('horizontal', False, dict(minspan=10, useblit=True)), ('vertical', True, dict(button=1)), ('horizontal', False, dict(props=dict(fill=True))), ('horizontal', False, dict(interactive=True))])
def test_span_selector(ax, orientation, onmove_callback, kwargs):
    onselect = mock.Mock(spec=noop, return_value=None)
    onmove = mock.Mock(spec=noop, return_value=None)
    if onmove_callback:
        kwargs['onmove_callback'] = onmove
    ax.set_aspect('auto')
    tax = ax.twinx()
    tool = widgets.SpanSelector(ax, onselect, orientation, **kwargs)
    do_event(tool, 'press', xdata=100, ydata=100, button=1)
    do_event(tool, 'onmove', xdata=199, ydata=199, button=1)
    do_event(tool, 'release', xdata=250, ydata=250, button=1)
    onselect.assert_called_once_with(100, 199)
    if onmove_callback:
        onmove.assert_called_once_with(100, 199)