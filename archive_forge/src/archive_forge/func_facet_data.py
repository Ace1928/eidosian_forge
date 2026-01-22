from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import (
from .palettes import color_palette, blend_palette
from ._docstrings import (
def facet_data(self):
    """Generator for name indices and data subsets for each facet.

        Yields
        ------
        (i, j, k), data_ijk : tuple of ints, DataFrame
            The ints provide an index into the {row, col, hue}_names attribute,
            and the dataframe contains a subset of the full data corresponding
            to each facet. The generator yields subsets that correspond with
            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`
            is None.

        """
    data = self.data
    if self.row_names:
        row_masks = [data[self._row_var] == n for n in self.row_names]
    else:
        row_masks = [np.repeat(True, len(self.data))]
    if self.col_names:
        col_masks = [data[self._col_var] == n for n in self.col_names]
    else:
        col_masks = [np.repeat(True, len(self.data))]
    if self.hue_names:
        hue_masks = [data[self._hue_var] == n for n in self.hue_names]
    else:
        hue_masks = [np.repeat(True, len(self.data))]
    for (i, row), (j, col), (k, hue) in product(enumerate(row_masks), enumerate(col_masks), enumerate(hue_masks)):
        data_ijk = data[row & col & hue & self._not_na]
        yield ((i, j, k), data_ijk)