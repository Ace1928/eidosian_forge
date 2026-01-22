from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
@property
def cursor_position_row(self):
    """
        Current row. (0-based.)
        """
    row, _ = self._find_line_start_index(self.cursor_position)
    return row