import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def on_selection(self, callback, append=False):
    """
        Register function to be called when the user selects one or more
        points in this trace.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 4 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.BoxSelector or plotly.callbacks.LassoSelector

        append : bool
            If False (the default), this callback replaces any previously
            defined on_selection callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points
        >>> points = Points()

        >>> def selection_fn(trace, points, selector):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_selection(selection_fn)

        Note: The creation of the `points` object is optional,
        it's simply a convenience to help the text editor perform completion
        on the `points` arguments inside `selection_fn`
        """
    if not append:
        del self._select_callbacks[:]
    if callback:
        self._select_callbacks.append(callback)