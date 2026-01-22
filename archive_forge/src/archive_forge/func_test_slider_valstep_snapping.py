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
def test_slider_valstep_snapping():
    fig, ax = plt.subplots()
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0, valinit=11.4, valstep=1)
    assert slider.val == 11
    slider = widgets.Slider(ax=ax, label='', valmin=0.0, valmax=24.0, valinit=11.4, valstep=[0, 1, 5.5, 19.7])
    assert slider.val == 5.5