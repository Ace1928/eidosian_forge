import sys
import re
import types
import unicodedata
from docutils import utils
from docutils.utils.error_reporting import ErrorOutput
def trim_end(self, n=1):
    """
        Remove items from the end of the list, without touching the parent.
        """
    if n > len(self.data):
        raise IndexError("Size of trim too large; can't trim %s items from a list of size %s." % (n, len(self.data)))
    elif n < 0:
        raise IndexError('Trim size must be >= 0.')
    del self.data[-n:]
    del self.items[-n:]