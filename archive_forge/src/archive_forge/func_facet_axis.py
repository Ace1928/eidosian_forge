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
def facet_axis(self, row_i, col_j, modify_state=True):
    """Make the axis identified by these indices active and return it."""
    if self._col_wrap is not None:
        ax = self.axes.flat[col_j]
    else:
        ax = self.axes[row_i, col_j]
    if modify_state:
        plt.sca(ax)
    return ax