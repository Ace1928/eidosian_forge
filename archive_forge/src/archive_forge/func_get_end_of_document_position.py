from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def get_end_of_document_position(self):
    """ Relative position for the end of the document. """
    return len(self.text) - self.cursor_position