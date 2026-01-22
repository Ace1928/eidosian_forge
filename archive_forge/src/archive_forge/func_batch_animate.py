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
@contextmanager
def batch_animate(self, duration=500, easing='cubic-in-out'):
    """
        Context manager to animate trace / layout updates

        Parameters
        ----------
        duration : number
            The duration of the transition, in milliseconds.
            If equal to zero, updates are synchronous.
        easing : string
            The easing function used for the transition.
            One of:
                - linear
                - quad
                - cubic
                - sin
                - exp
                - circle
                - elastic
                - back
                - bounce
                - linear-in
                - quad-in
                - cubic-in
                - sin-in
                - exp-in
                - circle-in
                - elastic-in
                - back-in
                - bounce-in
                - linear-out
                - quad-out
                - cubic-out
                - sin-out
                - exp-out
                - circle-out
                - elastic-out
                - back-out
                - bounce-out
                - linear-in-out
                - quad-in-out
                - cubic-in-out
                - sin-in-out
                - exp-in-out
                - circle-in-out
                - elastic-in-out
                - back-in-out
                - bounce-in-out

        Examples
        --------
        Suppose we have a figure widget, `fig`, with a single trace.

        >>> import plotly.graph_objs as go
        >>> fig = go.FigureWidget(data=[{'y': [3, 4, 2]}])

        1) Animate a change in the xaxis and yaxis ranges using default
        duration and easing parameters.

        >>> with fig.batch_animate():
        ...     fig.layout.xaxis.range = [0, 5]
        ...     fig.layout.yaxis.range = [0, 10]

        2) Animate a change in the size and color of the trace's markers
        over 2 seconds using the elastic-in-out easing method

        >>> with fig.batch_animate(duration=2000, easing='elastic-in-out'):
        ...     fig.data[0].marker.color = 'green'
        ...     fig.data[0].marker.size = 20
        """
    duration = self._animation_duration_validator.validate_coerce(duration)
    easing = self._animation_easing_validator.validate_coerce(easing)
    if self._in_batch_mode is True:
        yield
    else:
        try:
            self._in_batch_mode = True
            yield
        finally:
            self._in_batch_mode = False
            self._perform_batch_animate({'transition': {'duration': duration, 'easing': easing}, 'frame': {'duration': duration}})