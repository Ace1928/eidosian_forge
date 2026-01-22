import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def xitems(self):
    """Return iterator yielding (source, offset, value) tuples."""
    for value, (source, offset) in zip(self.data, self.items):
        yield (source, offset, value)