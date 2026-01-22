from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def get_strrow(self, row_num: int) -> str:
    """Get string representation of the row."""
    row = self.strrows[row_num]
    is_multicol = row_num < self.column_levels and self.fmt.header and self.multicolumn
    is_multirow = row_num >= self.header_levels and self.fmt.index and self.multirow and (self.index_levels > 1)
    is_cline_maybe_required = is_multirow and row_num < len(self.strrows) - 1
    crow = self._preprocess_row(row)
    if is_multicol:
        crow = self._format_multicolumn(crow)
    if is_multirow:
        crow = self._format_multirow(crow, row_num)
    lst = []
    lst.append(' & '.join(crow))
    lst.append(' \\\\')
    if is_cline_maybe_required:
        cline = self._compose_cline(row_num, len(self.strcols))
        lst.append(cline)
    return ''.join(lst)