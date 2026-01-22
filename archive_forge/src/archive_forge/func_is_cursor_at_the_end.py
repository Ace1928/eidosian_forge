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
def is_cursor_at_the_end(self):
    """ True when the cursor is at the end of the text. """
    return self.cursor_position == len(self.text)