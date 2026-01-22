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
def print_grid(self):
    """
        Print a visual layout of the figure's axes arrangement.
        This is only valid for figures that are created
        with plotly.tools.make_subplots.
        """
    if self._grid_str is None:
        raise Exception('Use plotly.tools.make_subplots to create a subplot grid.')
    print(self._grid_str)