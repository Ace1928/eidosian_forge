from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
def label_axis(axis, ticks=True, ticklabels=True, label=None, label_fontsize=None, tick_fontsize=None, ticklabel_rotation=None, ticklabel_horizontal_alignment=None, ticklabel_vertical_alignment=None):
    """Set axis ticks and labels.

    Parameters
    ----------
    axis : matplotlib.axis.{X,Y}Axis, mpl_toolkits.mplot3d.axis3d.{X,Y,Z}Axis
        Axis on which to draw labels and ticks
    ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks.
        If False, removes axis ticks.
        If a list, sets custom axis ticks
    ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels.
        If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    label : str or None (default : None)
        Axis labels. If None, no label is set.
    label_fontsize : str or None (default: None)
        Axis label font size.
    tick_fontsize : str or None (default: None)
        Axis tick label font size.
    ticklabel_rotation : int or None (default: None)
        Angle of rotation for tick labels
    ticklabel_horizontal_alignment : str or None (default: None)
        Horizontal alignment of tick labels
    ticklabel_vertical_alignment : str or None (default: None)
        Vertical alignment of tick labels
    """
    if ticks is False or ticks is None:
        axis.set_ticks([])
    elif ticks is True:
        pass
    else:
        axis.set_ticks(ticks)
    if ticklabels is False or ticklabels is None:
        axis.set_ticklabels([])
    else:
        tick_fontsize = parse_fontsize(tick_fontsize, 'large')
        if ticklabels is not True:
            axis.set_ticklabels(ticklabels)
        for tick in axis.get_ticklabels():
            if ticklabel_rotation is not None:
                tick.set_rotation(ticklabel_rotation)
            if ticklabel_horizontal_alignment is not None:
                tick.set_ha(ticklabel_horizontal_alignment)
            if ticklabel_vertical_alignment is not None:
                tick.set_va(ticklabel_vertical_alignment)
            tick.set_fontsize(tick_fontsize)
    if label is not None:
        label_fontsize = parse_fontsize(label_fontsize, 'x-large')
        axis.set_label_text(label, fontsize=label_fontsize)