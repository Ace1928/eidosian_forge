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
def leading_whitespace_in_current_line(self):
    """ The leading whitespace in the left margin of the current line.  """
    current_line = self.current_line
    length = len(current_line) - len(current_line.lstrip())
    return current_line[:length]