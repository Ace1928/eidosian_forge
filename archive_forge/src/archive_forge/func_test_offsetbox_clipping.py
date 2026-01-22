from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
@image_comparison(['offsetbox_clipping'], remove_text=True)
def test_offsetbox_clipping():
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    assert da.clip_children
    bg = mpatches.Rectangle((0, 0), size, size, facecolor='#CCCCCC', edgecolor='None', linewidth=0)
    line = mlines.Line2D([-size * 0.5, size * 1.5], [size / 2, size / 2], color='black', linewidth=10)
    anchored_box = AnchoredOffsetbox(loc='center', child=da, pad=0.0, frameon=False, bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes, borderpad=0.0)
    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))