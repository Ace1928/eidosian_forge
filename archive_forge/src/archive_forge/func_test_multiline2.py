from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
@image_comparison(['multiline2'], style='mpl20')
def test_multiline2():
    plt.rcParams['text.kerning_factor'] = 6
    fig, ax = plt.subplots()
    ax.set_xlim([0, 1.4])
    ax.set_ylim([0, 2])
    ax.axhline(0.5, color='C2', linewidth=0.3)
    sts = ['Line', '2 Lineg\n 2 Lg', '$\\sum_i x $', 'hi $\\sum_i x $\ntest', 'test\n $\\sum_i x $', '$\\sum_i x $\n $\\sum_i x $']
    renderer = fig.canvas.get_renderer()

    def draw_box(ax, tt):
        r = mpatches.Rectangle((0, 0), 1, 1, clip_on=False, transform=ax.transAxes)
        r.set_bounds(tt.get_window_extent(renderer).transformed(ax.transAxes.inverted()).bounds)
        ax.add_patch(r)
    horal = 'left'
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 0.5, st, horizontalalignment=horal, verticalalignment='bottom')
        draw_box(ax, tt)
    ax.text(1.2, 0.5, 'Bottom align', color='C2')
    ax.axhline(1.3, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.3, st, horizontalalignment=horal, verticalalignment='top')
        draw_box(ax, tt)
    ax.text(1.2, 1.3, 'Top align', color='C2')
    ax.axhline(1.8, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.8, st, horizontalalignment=horal, verticalalignment='baseline')
        draw_box(ax, tt)
    ax.text(1.2, 1.8, 'Baseline align', color='C2')
    ax.axhline(0.1, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 0.1, st, horizontalalignment=horal, verticalalignment='bottom', rotation=20)
        draw_box(ax, tt)
    ax.text(1.2, 0.1, 'Bot align, rot20', color='C2')