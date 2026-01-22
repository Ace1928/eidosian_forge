from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def end_of_paragraph(self, count=1, after=False):
    """
        Return the end of the current paragraph. (Relative cursor position.)
        """

    def match_func(text):
        return not text or text.isspace()
    line_index = self.find_next_matching_line(match_func=match_func, count=count)
    if line_index:
        add = 0 if after else 1
        return max(0, self.get_cursor_down_position(count=line_index) - add)
    else:
        return len(self.text_after_cursor)