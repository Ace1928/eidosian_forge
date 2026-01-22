from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def empty_line_count_at_the_end(self):
    """
        Return number of empty lines at the end of the document.
        """
    count = 0
    for line in self.lines[::-1]:
        if not line or line.isspace():
            count += 1
        else:
            break
    return count