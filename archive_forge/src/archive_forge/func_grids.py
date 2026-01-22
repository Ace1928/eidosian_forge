import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def grids(fig=None, value='solid'):
    """Sets the value of the grid_lines for the axis to the passed value.
    The default value is `solid`.

    Parameters
    ----------
    fig: Figure or None(default: None)
        The figure for which the axes should be edited. If the value is None,
        the current figure is used.
    value: {'none', 'solid', 'dashed'}
        The display of the grid_lines
    """
    if fig is None:
        fig = current_figure()
    for a in fig.axes:
        a.grid_lines = value