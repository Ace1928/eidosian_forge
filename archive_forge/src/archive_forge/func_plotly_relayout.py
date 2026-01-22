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
def plotly_relayout(self, relayout_data, **kwargs):
    """
        Perform a Plotly relayout operation on the figure's layout

        Parameters
        ----------
        relayout_data : dict
            Dict of layout updates

            dict keys are strings that specify the properties to be updated.
            Nested properties are expressed by joining successive keys on
            '.' characters (e.g. 'xaxis.range')

            dict values are the values to use to update the layout.

        Returns
        -------
        None
        """
    if 'source_view_id' in kwargs:
        msg_kwargs = {'source_view_id': kwargs['source_view_id']}
    else:
        msg_kwargs = {}
    relayout_changes = self._perform_plotly_relayout(relayout_data)
    if relayout_changes:
        self._send_relayout_msg(relayout_changes, **msg_kwargs)
        self._dispatch_layout_change_callbacks(relayout_changes)