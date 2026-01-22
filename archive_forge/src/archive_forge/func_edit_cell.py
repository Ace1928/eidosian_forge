import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def edit_cell(self, index, column, value):
    """
        Edit a cell of the grid, given the index and column of the cell
        to edit, as well as the new value of the cell. Results in a
        ``cell_edited`` event being fired.

        Parameters
        ----------
        index : object
            The index of the row containing the cell that is to be edited.
        column : str
            The name of the column containing the cell that is to be edited.
        value : object
            The new value for the cell.
        """
    old_value = self._df.loc[index, column]
    self._df.loc[index, column] = value
    self._unfiltered_df.loc[index, column] = value
    self._update_table(triggered_by='edit_cell', fire_data_change_event=True)
    self._notify_listeners({'name': 'cell_edited', 'index': index, 'column': column, 'old': old_value, 'new': value, 'source': 'api'})