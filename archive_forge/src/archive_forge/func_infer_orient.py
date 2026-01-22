from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
def infer_orient(x=None, y=None, orient=None, require_numeric=True):
    """Determine how the plot should be oriented based on the data.

    For historical reasons, the convention is to call a plot "horizontally"
    or "vertically" oriented based on the axis representing its dependent
    variable. Practically, this is used when determining the axis for
    numerical aggregation.

    Parameters
    ----------
    x, y : Vector data or None
        Positional data vectors for the plot.
    orient : string or None
        Specified orientation. If not None, can be "x" or "y", or otherwise
        must start with "v" or "h".
    require_numeric : bool
        If set, raise when the implied dependent variable is not numeric.

    Returns
    -------
    orient : "x" or "y"

    Raises
    ------
    ValueError: When `orient` is an unknown string.
    TypeError: When dependent variable is not numeric, with `require_numeric`

    """
    x_type = None if x is None else variable_type(x)
    y_type = None if y is None else variable_type(y)
    nonnumeric_dv_error = '{} orientation requires numeric `{}` variable.'
    single_var_warning = '{} orientation ignored with only `{}` specified.'
    if x is None:
        if str(orient).startswith('h'):
            warnings.warn(single_var_warning.format('Horizontal', 'y'))
        if require_numeric and y_type != 'numeric':
            raise TypeError(nonnumeric_dv_error.format('Vertical', 'y'))
        return 'x'
    elif y is None:
        if str(orient).startswith('v'):
            warnings.warn(single_var_warning.format('Vertical', 'x'))
        if require_numeric and x_type != 'numeric':
            raise TypeError(nonnumeric_dv_error.format('Horizontal', 'x'))
        return 'y'
    elif str(orient).startswith('v') or orient == 'x':
        if require_numeric and y_type != 'numeric':
            raise TypeError(nonnumeric_dv_error.format('Vertical', 'y'))
        return 'x'
    elif str(orient).startswith('h') or orient == 'y':
        if require_numeric and x_type != 'numeric':
            raise TypeError(nonnumeric_dv_error.format('Horizontal', 'x'))
        return 'y'
    elif orient is not None:
        err = f"`orient` must start with 'v' or 'h' or be None, but `{repr(orient)}` was passed."
        raise ValueError(err)
    elif x_type != 'categorical' and y_type == 'categorical':
        return 'y'
    elif x_type != 'numeric' and y_type == 'numeric':
        return 'x'
    elif x_type == 'numeric' and y_type != 'numeric':
        return 'y'
    elif require_numeric and 'numeric' not in (x_type, y_type):
        err = 'Neither the `x` nor `y` variable appears to be numeric.'
        raise TypeError(err)
    else:
        return 'x'