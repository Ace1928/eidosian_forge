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
def test_offsetbox_clip_children():
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    bg = mpatches.Rectangle((0, 0), size, size, facecolor='#CCCCCC', edgecolor='None', linewidth=0)
    line = mlines.Line2D([-size * 0.5, size * 1.5], [size / 2, size / 2], color='black', linewidth=10)
    anchored_box = AnchoredOffsetbox(loc='center', child=da, pad=0.0, frameon=False, bbox_to_anchor=(0.5, 0.5), bbox_transform=ax.transAxes, borderpad=0.0)
    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)
    fig.canvas.draw()
    assert not fig.stale
    da.clip_children = True
    assert fig.stale