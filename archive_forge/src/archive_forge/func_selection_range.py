from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def selection_range(self):
    """
        Return (from, to) tuple of the selection.
        start and end position are included.

        This doesn't take the selection type into account. Use
        `selection_ranges` instead.
        """
    if self.selection:
        from_, to = sorted([self.cursor_position, self.selection.original_cursor_position])
    else:
        from_, to = (self.cursor_position, self.cursor_position)
    return (from_, to)