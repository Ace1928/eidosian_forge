import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def plot_children(fig, lg=None, level=0):
    """Simple plotting to show where boxes are."""
    if lg is None:
        _layoutgrids = fig.get_layout_engine().execute(fig)
        lg = _layoutgrids[fig]
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    col = colors[level]
    for i in range(lg.nrows):
        for j in range(lg.ncols):
            bb = lg.get_outer_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bb.p0, bb.width, bb.height, linewidth=1, edgecolor='0.7', facecolor='0.7', alpha=0.2, transform=fig.transFigure, zorder=-3))
            bbi = lg.get_inner_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=2, edgecolor=col, facecolor='none', transform=fig.transFigure, zorder=-2))
            bbi = lg.get_left_margin_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0, edgecolor='none', alpha=0.2, facecolor=[0.5, 0.7, 0.5], transform=fig.transFigure, zorder=-2))
            bbi = lg.get_right_margin_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0, edgecolor='none', alpha=0.2, facecolor=[0.7, 0.5, 0.5], transform=fig.transFigure, zorder=-2))
            bbi = lg.get_bottom_margin_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0, edgecolor='none', alpha=0.2, facecolor=[0.5, 0.5, 0.7], transform=fig.transFigure, zorder=-2))
            bbi = lg.get_top_margin_bbox(rows=i, cols=j)
            fig.add_artist(mpatches.Rectangle(bbi.p0, bbi.width, bbi.height, linewidth=0, edgecolor='none', alpha=0.2, facecolor=[0.7, 0.2, 0.7], transform=fig.transFigure, zorder=-2))
    for ch in lg.children.flat:
        if ch is not None:
            plot_children(fig, ch, level=level + 1)