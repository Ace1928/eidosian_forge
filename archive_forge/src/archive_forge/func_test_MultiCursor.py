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
@pytest.mark.parametrize('horizOn', [False, True])
@pytest.mark.parametrize('vertOn', [False, True])
def test_MultiCursor(horizOn, vertOn):
    ax1, ax3 = plt.figure().subplots(2, sharex=True)
    ax2 = plt.figure().subplots()
    multi = widgets.MultiCursor(None, (ax1, ax2), useblit=False, horizOn=horizOn, vertOn=vertOn)
    assert len(multi.vlines) == 2
    assert len(multi.hlines) == 2
    event = mock_event(ax1, xdata=0.5, ydata=0.25)
    multi.onmove(event)
    ax1.figure.canvas.draw()
    for l in multi.vlines:
        assert l.get_xdata() == (0.5, 0.5)
    for l in multi.hlines:
        assert l.get_ydata() == (0.25, 0.25)
    assert len([line for line in multi.vlines if line.get_visible()]) == (2 if vertOn else 0)
    assert len([line for line in multi.hlines if line.get_visible()]) == (2 if horizOn else 0)
    multi.horizOn = not multi.horizOn
    multi.vertOn = not multi.vertOn
    event = mock_event(ax1, xdata=0.5, ydata=0.25)
    multi.onmove(event)
    assert len([line for line in multi.vlines if line.get_visible()]) == (0 if vertOn else 2)
    assert len([line for line in multi.hlines if line.get_visible()]) == (0 if horizOn else 2)
    event = mock_event(ax3, xdata=0.75, ydata=0.75)
    multi.onmove(event)
    for l in multi.vlines:
        assert l.get_xdata() == (0.5, 0.5)
    for l in multi.hlines:
        assert l.get_ydata() == (0.25, 0.25)