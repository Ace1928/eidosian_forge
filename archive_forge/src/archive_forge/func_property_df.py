from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
@property
def property_df(self):
    """ A pandas dataframe of all of the properties of the model with their
        values, types, and docstrings.  The base information for the datatable.

        """
    return self._prop_df