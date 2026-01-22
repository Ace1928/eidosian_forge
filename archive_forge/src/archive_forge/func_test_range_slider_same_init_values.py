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
def test_range_slider_same_init_values(orientation):
    if orientation == 'vertical':
        idx = [1, 0, 3, 2]
    else:
        idx = [0, 1, 2, 3]
    fig, ax = plt.subplots()
    slider = widgets.RangeSlider(ax=ax, label='', valmin=0.0, valmax=1.0, orientation=orientation, valinit=[0, 0])
    box = slider.poly.get_extents().transformed(ax.transAxes.inverted())
    assert_allclose(box.get_points().flatten()[idx], [0, 0.25, 0, 0.75])