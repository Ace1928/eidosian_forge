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
def current_line_after_cursor(self):
    """ Text from the cursor until the end of the line. """
    text, _, _ = self.text_after_cursor.partition('\n')
    return text