from __future__ import annotations
from math import ceil
from typing import TYPE_CHECKING
import warnings
from matplotlib import ticker
import matplotlib.table
import numpy as np
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
def handle_shared_axes(axarr: Iterable[Axes], nplots: int, naxes: int, nrows: int, ncols: int, sharex: bool, sharey: bool) -> None:
    if nplots > 1:
        row_num = lambda x: x.get_subplotspec().rowspan.start
        col_num = lambda x: x.get_subplotspec().colspan.start
        is_first_col = lambda x: x.get_subplotspec().is_first_col()
        if nrows > 1:
            try:
                layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
                for ax in axarr:
                    layout[row_num(ax), col_num(ax)] = ax.get_visible()
                for ax in axarr:
                    if not layout[row_num(ax) + 1, col_num(ax)]:
                        continue
                    if sharex or _has_externally_shared_axis(ax, 'x'):
                        _remove_labels_from_axis(ax.xaxis)
            except IndexError:
                is_last_row = lambda x: x.get_subplotspec().is_last_row()
                for ax in axarr:
                    if is_last_row(ax):
                        continue
                    if sharex or _has_externally_shared_axis(ax, 'x'):
                        _remove_labels_from_axis(ax.xaxis)
        if ncols > 1:
            for ax in axarr:
                if is_first_col(ax):
                    continue
                if sharey or _has_externally_shared_axis(ax, 'y'):
                    _remove_labels_from_axis(ax.yaxis)