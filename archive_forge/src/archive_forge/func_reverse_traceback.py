import sys
import traceback
from mako import compat
from mako import util
@property
def reverse_traceback(self):
    """Return the same data as traceback, except in reverse order."""
    return list(self._get_reformatted_records(self.reverse_records))