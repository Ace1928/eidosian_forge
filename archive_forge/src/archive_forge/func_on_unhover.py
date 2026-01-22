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
def on_unhover(self, callback, append=False):
    """
        Register function to be called when the user unhovers away from one
        or more points in this trace.

        Note: Callbacks will only be triggered when the trace belongs to a
        instance of plotly.graph_objs.FigureWidget and it is displayed in an
        ipywidget context. Callbacks will not be triggered on figures
        that are displayed using plot/iplot.

        Parameters
        ----------
        callback
            Callable function that accepts 3 arguments

            - this trace
            - plotly.callbacks.Points object
            - plotly.callbacks.InputDeviceState object

        append : bool
            If False (the default), this callback replaces any previously
            defined on_unhover callbacks for this trace. If True,
            this callback is appended to the list of any previously defined
            callbacks.

        Returns
        -------
        None

        Examples
        --------

        >>> import plotly.graph_objects as go
        >>> from plotly.callbacks import Points, InputDeviceState
        >>> points, state = Points(), InputDeviceState()

        >>> def unhover_fn(trace, points, state):
        ...     inds = points.point_inds
        ...     # Do something

        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])
        >>> trace.on_unhover(unhover_fn)

        Note: The creation of the `points` and `state` objects is optional,
        it's simply a convenience to help the text editor perform completion
        on the arguments inside `unhover_fn`
        """
    if not append:
        del self._unhover_callbacks[:]
    if callback:
        self._unhover_callbacks.append(callback)