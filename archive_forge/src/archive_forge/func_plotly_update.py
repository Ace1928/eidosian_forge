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
def plotly_update(self, restyle_data=None, relayout_data=None, trace_indexes=None, **kwargs):
    """
        Perform a Plotly update operation on the figure.

        Note: This operation both mutates and returns the figure

        Parameters
        ----------
        restyle_data : dict
            Traces update specification. See the docstring for the
            `plotly_restyle` method for details
        relayout_data : dict
            Layout update specification. See the docstring for the
            `plotly_relayout` method for details
        trace_indexes :
            Trace index, or list of trace indexes, that the update operation
            applies to. Defaults to all trace indexes.

        Returns
        -------
        BaseFigure
            None
        """
    if 'source_view_id' in kwargs:
        msg_kwargs = {'source_view_id': kwargs['source_view_id']}
    else:
        msg_kwargs = {}
    restyle_changes, relayout_changes, trace_indexes = self._perform_plotly_update(restyle_data=restyle_data, relayout_data=relayout_data, trace_indexes=trace_indexes)
    if restyle_changes or relayout_changes:
        self._send_update_msg(restyle_data=restyle_changes, relayout_data=relayout_changes, trace_indexes=trace_indexes, **msg_kwargs)
    if restyle_changes:
        self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)
    if relayout_changes:
        self._dispatch_layout_change_callbacks(relayout_changes)