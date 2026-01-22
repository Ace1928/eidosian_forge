from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
Set axis ticks and labels.

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
    