from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_backwards(self, sub, in_current_line=False, ignore_case=False, count=1):
    """
        Find `text` before the cursor, return position relative to the cursor
        position. Return `None` if nothing was found.

        :param count: Find the n-th occurance.
        """
    if in_current_line:
        before_cursor = self.current_line_before_cursor[::-1]
    else:
        before_cursor = self.text_before_cursor[::-1]
    flags = re.IGNORECASE if ignore_case else 0
    iterator = re.finditer(re.escape(sub[::-1]), before_cursor, flags)
    try:
        for i, match in enumerate(iterator):
            if i + 1 == count:
                return -match.start(0) - len(sub)
    except StopIteration:
        pass