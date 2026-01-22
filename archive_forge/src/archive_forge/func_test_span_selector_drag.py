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
@pytest.mark.parametrize('drag_from_anywhere', [True, False])
def test_span_selector_drag(ax, drag_from_anywhere):
    tool = widgets.SpanSelector(ax, onselect=noop, direction='horizontal', interactive=True, drag_from_anywhere=drag_from_anywhere)
    click_and_drag(tool, start=(10, 10), end=(100, 120))
    assert tool.extents == (10, 100)
    click_and_drag(tool, start=(25, 15), end=(35, 25))
    if drag_from_anywhere:
        assert tool.extents == (20, 110)
    else:
        assert tool.extents == (25, 35)
    click_and_drag(tool, start=(175, 185), end=(185, 195))
    assert tool.extents == (175, 185)