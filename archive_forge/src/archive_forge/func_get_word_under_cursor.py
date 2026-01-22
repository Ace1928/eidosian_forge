from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_word_under_cursor(self, WORD=False):
    """
        Return the word, currently below the cursor.
        This returns an empty string when the cursor is on a whitespace region.
        """
    start, end = self.find_boundaries_of_current_word(WORD=WORD)
    return self.text[self.cursor_position + start:self.cursor_position + end]