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
def test_save_blitted_widget_as_pdf():
    from matplotlib.widgets import CheckButtons, RadioButtons
    from matplotlib.cbook import _get_running_interactive_framework
    if _get_running_interactive_framework() not in ['headless', None]:
        pytest.xfail('Callback exceptions are not raised otherwise.')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 2), width_ratios=[1, 2])
    default_rb = RadioButtons(ax[0, 0], ['Apples', 'Oranges'])
    styled_rb = RadioButtons(ax[0, 1], ['Apples', 'Oranges'], label_props={'color': ['red', 'orange'], 'fontsize': [16, 20]}, radio_props={'edgecolor': ['red', 'orange'], 'facecolor': ['mistyrose', 'peachpuff']})
    default_cb = CheckButtons(ax[1, 0], ['Apples', 'Oranges'], actives=[True, True])
    styled_cb = CheckButtons(ax[1, 1], ['Apples', 'Oranges'], actives=[True, True], label_props={'color': ['red', 'orange'], 'fontsize': [16, 20]}, frame_props={'edgecolor': ['red', 'orange'], 'facecolor': ['mistyrose', 'peachpuff']}, check_props={'color': ['darkred', 'darkorange']})
    ax[0, 0].set_title('Default')
    ax[0, 1].set_title('Stylized')
    fig.canvas.draw()
    with io.BytesIO() as result_after:
        fig.savefig(result_after, format='pdf')