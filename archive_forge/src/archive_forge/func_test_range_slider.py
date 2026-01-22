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
@pytest.mark.parametrize('orientation', ['horizontal', 'vertical'])
def test_range_slider(orientation):
    if orientation == 'vertical':
        idx = [1, 0, 3, 2]
    else:
        idx = [0, 1, 2, 3]
    fig, ax = plt.subplots()
    slider = widgets.RangeSlider(ax=ax, label='', valmin=0.0, valmax=1.0, orientation=orientation, valinit=[0.1, 0.34])
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.get_points().flatten()[idx], [0.1, 0.25, 0.34, 0.75])
    assert_allclose(slider.val, (0.1, 0.34))

    def handle_positions(slider):
        if orientation == 'vertical':
            return [h.get_ydata()[0] for h in slider._handles]
        else:
            return [h.get_xdata()[0] for h in slider._handles]
    slider.set_val((0.4, 0.6))
    assert_allclose(slider.val, (0.4, 0.6))
    assert_allclose(handle_positions(slider), (0.4, 0.6))
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.get_points().flatten()[idx], [0.4, 0.25, 0.6, 0.75])
    slider.set_val((0.2, 0.1))
    assert_allclose(slider.val, (0.1, 0.2))
    assert_allclose(handle_positions(slider), (0.1, 0.2))
    slider.set_val((-1, 10))
    assert_allclose(slider.val, (0, 1))
    assert_allclose(handle_positions(slider), (0, 1))
    slider.reset()
    assert_allclose(slider.val, (0.1, 0.34))
    assert_allclose(handle_positions(slider), (0.1, 0.34))