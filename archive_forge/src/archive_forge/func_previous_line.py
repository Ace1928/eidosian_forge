import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def previous_line(self, n=1):
    """Load `self.line` with the `n`'th previous line and return it."""
    self.line_offset -= n
    if self.line_offset < 0:
        self.line = None
    else:
        self.line = self.input_lines[self.line_offset]
    self.notify_observers()
    return self.line