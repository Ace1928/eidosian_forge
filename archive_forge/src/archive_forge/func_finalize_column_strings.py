import collections
import enum
import itertools as it
from typing import DefaultDict, List, Optional, Tuple
from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
def finalize_column_strings(self, column_strings, col_widths):
    best_values = [-1 for _ in column_strings]
    if self._colorize == Colorize.ROWWISE:
        row_min = min((r.median for r in self._results if r is not None))
        best_values = [row_min for _ in column_strings]
    elif self._colorize == Colorize.COLUMNWISE:
        best_values = [optional_min((r.median for r in column.get_results_for(self._row_group) if r is not None)) for column in self._columns or ()]
    row_contents = [column_strings[0].ljust(col_widths[0])]
    for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values):
        col_str = col_str.center(width)
        if self._colorize != Colorize.NONE and result is not None and (best_value is not None):
            col_str = self.color_segment(col_str, result.median, best_value)
        row_contents.append(col_str)
    return row_contents