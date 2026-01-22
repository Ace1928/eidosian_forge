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
@pytest.mark.parametrize('interactive', [True, False])
def test_span_selector_onselect(ax, interactive):
    onselect = mock.Mock(spec=noop, return_value=None)
    tool = widgets.SpanSelector(ax, onselect, 'horizontal', interactive=interactive)
    click_and_drag(tool, start=(100, 100), end=(150, 100))
    onselect.assert_called_once()
    assert tool.extents == (100, 150)
    onselect.reset_mock()
    click_and_drag(tool, start=(10, 100), end=(10, 100))
    onselect.assert_called_once()