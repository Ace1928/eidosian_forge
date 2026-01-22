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
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_span_selector_animated_artists_callback():
    """Check that the animated artists changed in callbacks are updated."""
    x = np.linspace(0, 2 * np.pi, 100)
    values = np.sin(x)
    fig, ax = plt.subplots()
    ln, = ax.plot(x, values, animated=True)
    ln2, = ax.plot([], animated=True)
    plt.pause(0.1)
    ax.draw_artist(ln)
    fig.canvas.blit(fig.bbox)

    def mean(vmin, vmax):
        indmin, indmax = np.searchsorted(x, (vmin, vmax))
        v = values[indmin:indmax].mean()
        ln2.set_data(x, np.full_like(x, v))
    span = widgets.SpanSelector(ax, mean, direction='horizontal', onmove_callback=mean, interactive=True, drag_from_anywhere=True, useblit=True)
    press_data = [1, 2]
    move_data = [2, 2]
    do_event(span, 'press', xdata=press_data[0], ydata=press_data[1], button=1)
    do_event(span, 'onmove', xdata=move_data[0], ydata=move_data[1], button=1)
    assert span._get_animated_artists() == (ln, ln2)
    assert ln.stale is False
    assert ln2.stale
    assert_allclose(ln2.get_ydata(), 0.9547335049088455)
    span.update()
    assert ln2.stale is False
    press_data = [4, 0]
    move_data = [5, 2]
    release_data = [5, 2]
    do_event(span, 'press', xdata=press_data[0], ydata=press_data[1], button=1)
    do_event(span, 'onmove', xdata=move_data[0], ydata=move_data[1], button=1)
    assert ln.stale is False
    assert ln2.stale
    assert_allclose(ln2.get_ydata(), -0.9424150707548072)
    do_event(span, 'release', xdata=release_data[0], ydata=release_data[1], button=1)
    assert ln2.stale is False