from itertools import product
import io
import platform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import cbook
from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Ellipse
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.testing.decorators import (
from mpl_toolkits.axes_grid1 import (
from mpl_toolkits.axes_grid1.anchored_artists import (
from mpl_toolkits.axes_grid1.axes_divider import (
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
from mpl_toolkits.axes_grid1.inset_locator import (
import mpl_toolkits.axes_grid1.mpl_axes
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
@image_comparison(['fill_facecolor.png'], remove_text=True, style='mpl20')
def test_fill_facecolor():
    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(5, 5)
    for i in range(1, 4):
        ax[i].yaxis.set_visible(False)
    ax[4].yaxis.tick_right()
    bbox = Bbox.from_extents(0, 0.4, 1, 0.6)
    bbox1 = TransformedBbox(bbox, ax[0].transData)
    bbox2 = TransformedBbox(bbox, ax[1].transData)
    p = BboxConnectorPatch(bbox1, bbox2, loc1a=1, loc2a=2, loc1b=4, loc2b=3, ec='r', fc='b')
    p.set_clip_on(False)
    ax[0].add_patch(p)
    axins = zoomed_inset_axes(ax[0], 1, loc='upper right')
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0, 0.2)
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    mark_inset(ax[0], axins, loc1=2, loc2=4, fc='b', ec='0.5')
    bbox3 = TransformedBbox(bbox, ax[1].transData)
    bbox4 = TransformedBbox(bbox, ax[2].transData)
    p = BboxConnectorPatch(bbox3, bbox4, loc1a=1, loc2a=2, loc1b=4, loc2b=3, ec='r', facecolor='y')
    p.set_clip_on(False)
    ax[1].add_patch(p)
    axins = zoomed_inset_axes(ax[1], 1, loc='upper right')
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0, 0.2)
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    mark_inset(ax[1], axins, loc1=2, loc2=4, facecolor='y', ec='0.5')
    bbox5 = TransformedBbox(bbox, ax[2].transData)
    bbox6 = TransformedBbox(bbox, ax[3].transData)
    p = BboxConnectorPatch(bbox5, bbox6, loc1a=1, loc2a=2, loc1b=4, loc2b=3, ec='r', color='g')
    p.set_clip_on(False)
    ax[2].add_patch(p)
    axins = zoomed_inset_axes(ax[2], 1, loc='upper right')
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0, 0.2)
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    mark_inset(ax[2], axins, loc1=2, loc2=4, color='g', ec='0.5')
    bbox7 = TransformedBbox(bbox, ax[3].transData)
    bbox8 = TransformedBbox(bbox, ax[4].transData)
    p = BboxConnectorPatch(bbox7, bbox8, loc1a=1, loc2a=2, loc1b=4, loc2b=3, ec='r', fc='g', fill=False)
    p.set_clip_on(False)
    ax[3].add_patch(p)
    axins = zoomed_inset_axes(ax[3], 1, loc='upper right')
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0, 0.2)
    axins.xaxis.set_ticks([])
    axins.yaxis.set_ticks([])
    mark_inset(ax[3], axins, loc1=2, loc2=4, fc='g', ec='0.5', fill=False)