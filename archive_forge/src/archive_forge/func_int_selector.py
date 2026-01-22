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
def int_selector(func=None, trait='selected', **kwargs):
    """Creates a `FastIntervalSelector` interaction for the `figure`.

    Also attaches the function `func` as an event listener for the
    trait `trait`.

    Parameters
    ----------

    func: function
        The call back function. It should take at least two arguments. The name
        of the trait and the value of the trait are passed as arguments.
    trait: string
        The name of the IntervalSelector trait whose change triggers the
        call back function `func`.
    """
    return _create_selector(FastIntervalSelector, func, trait, **kwargs)