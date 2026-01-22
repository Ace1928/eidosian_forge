from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def find_next_word_ending(self, include_current_position=False, count=1, WORD=False):
    """
        Return an index relative to the cursor position pointing to the end
        of the next word. Return `None` if nothing was found.
        """
    if count < 0:
        return self.find_previous_word_ending(count=-count, WORD=WORD)
    if include_current_position:
        text = self.text_after_cursor
    else:
        text = self.text_after_cursor[1:]
    regex = _FIND_BIG_WORD_RE if WORD else _FIND_WORD_RE
    iterable = regex.finditer(text)
    try:
        for i, match in enumerate(iterable):
            if i + 1 == count:
                value = match.end(1)
                if include_current_position:
                    return value
                else:
                    return value + 1
    except StopIteration:
        pass