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
@check_figures_equal(extensions=['png'])
def test_check_buttons_lines(fig_test, fig_ref):
    cb = widgets.CheckButtons(fig_test.subplots(), ['', ''], [True, True])
    with pytest.warns(DeprecationWarning, match='The lines attribute was deprecated'):
        cb.lines
    for rectangle in cb._rectangles:
        rectangle.set_visible(False)
    ax = fig_ref.add_subplot(xticks=[], yticks=[])
    ys = [2 / 3, 1 / 3]
    dy = 1 / 3
    w, h = (dy / 2, dy / 2)
    lineparams = {'color': 'k', 'linewidth': 1.25, 'transform': ax.transAxes, 'solid_capstyle': 'butt'}
    for i, y in enumerate(ys):
        x, y = (0.05, y - h / 2)
        l1 = Line2D([x, x + w], [y + h, y], **lineparams)
        l2 = Line2D([x, x + w], [y, y + h], **lineparams)
        l1.set_visible(True)
        l2.set_visible(True)
        ax.add_line(l1)
        ax.add_line(l2)