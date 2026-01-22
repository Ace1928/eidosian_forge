from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def translate_row_col_to_index(self, row, col):
    """
        Given a (row, col) tuple, return the corresponding index.
        (Row and col params are 0-based.)

        Negative row/col values are turned into zero.
        """
    try:
        result = self._line_start_indexes[row]
        line = self.lines[row]
    except IndexError:
        if row < 0:
            result = self._line_start_indexes[0]
            line = self.lines[0]
        else:
            result = self._line_start_indexes[-1]
            line = self.lines[-1]
    result += max(0, min(col, len(line)))
    result = max(0, min(result, len(self.text)))
    return result