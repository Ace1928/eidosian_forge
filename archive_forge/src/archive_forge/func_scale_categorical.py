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
def scale_categorical(self, axis, order=None, formatter=None):
    """
        Enforce categorical (fixed-scale) rules for the data on given axis.

        Parameters
        ----------
        axis : "x" or "y"
            Axis of the plot to operate on.
        order : list
            Order that unique values should appear in.
        formatter : callable
            Function mapping values to a string representation.

        Returns
        -------
        self

        """
    _check_argument('axis', ['x', 'y'], axis)
    if axis not in self.variables:
        self.variables[axis] = None
        self.var_types[axis] = 'categorical'
        self.plot_data[axis] = ''
    if self.var_types[axis] == 'numeric':
        self.plot_data = self.plot_data.sort_values(axis, kind='mergesort')
    cat_data = self.plot_data[axis].dropna()
    self._var_ordered[axis] = order is not None or cat_data.dtype.name == 'category'
    order = pd.Index(categorical_order(cat_data, order), name=axis)
    if formatter is not None:
        cat_data = cat_data.map(formatter)
        order = order.map(formatter)
    else:
        cat_data = cat_data.astype(str)
        order = order.astype(str)
    self.var_levels[axis] = order
    self.var_types[axis] = 'categorical'
    self.plot_data[axis] = cat_data
    return self