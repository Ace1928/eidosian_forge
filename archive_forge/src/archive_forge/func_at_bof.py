import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def at_bof(self):
    """Return 1 if the input is at or before beginning-of-file."""
    return self.line_offset <= 0