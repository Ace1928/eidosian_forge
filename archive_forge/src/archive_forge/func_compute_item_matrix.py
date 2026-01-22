import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def compute_item_matrix(items, row_first: bool=False, empty=None, *, separator_size=2, displaywidth=80) -> Tuple[List[List[int]], Dict[str, int]]:
    """Returns a nested list, and info to columnize items

    Parameters
    ----------
    items
        list of strings to columize
    row_first : (default False)
        Whether to compute columns for a row-first matrix instead of
        column-first (default).
    empty : (default None)
        default value to fill list if needed
    separator_size : int (default=2)
        How much characters will be used as a separation between each columns.
    displaywidth : int (default=80)
        The width of the area onto which the columns should enter

    Returns
    -------
    strings_matrix
        nested list of string, the outer most list contains as many list as
        rows, the innermost lists have each as many element as columns. If the
        total number of elements in `items` does not equal the product of
        rows*columns, the last element of some lists are filled with `None`.
    dict_info
        some info to make columnize easier:

        num_columns
          number of columns
        max_rows
          maximum number of rows (final number may be less)
        column_widths
          list of with of each columns
        optimal_separator_width
          best separator width between columns

    Examples
    --------
    ::

        In [1]: l = ['aaa','b','cc','d','eeeee','f','g','h','i','j','k','l']
        In [2]: list, info = compute_item_matrix(l, displaywidth=12)
        In [3]: list
        Out[3]: [['aaa', 'f', 'k'], ['b', 'g', 'l'], ['cc', 'h', None], ['d', 'i', None], ['eeeee', 'j', None]]
        In [4]: ideal = {'num_columns': 3, 'column_widths': [5, 1, 1], 'optimal_separator_width': 2, 'max_rows': 5}
        In [5]: all((info[k] == ideal[k] for k in ideal.keys()))
        Out[5]: True
    """
    warnings.warn('`compute_item_matrix` is Pending Deprecation since IPython 8.17.It is considered fro removal in in future version. Please open an issue if you believe it should be kept.', stacklevel=2, category=PendingDeprecationWarning)
    info = _find_optimal(list(map(len, items)), row_first, separator_size=separator_size, displaywidth=displaywidth)
    nrow, ncol = (info['max_rows'], info['num_columns'])
    if row_first:
        return ([[_get_or_default(items, r * ncol + c, default=empty) for c in range(ncol)] for r in range(nrow)], info)
    else:
        return ([[_get_or_default(items, c * nrow + r, default=empty) for c in range(ncol)] for r in range(nrow)], info)