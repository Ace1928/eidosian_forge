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
def char_before_cursor(self):
    """ Return character before the cursor or an empty string. """
    return self._get_char_relative_to_cursor(-1) or ''