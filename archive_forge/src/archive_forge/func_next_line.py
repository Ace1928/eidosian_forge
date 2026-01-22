from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def next_line(self):
    """Get the next line without the newline or None on EOF."""
    line = self.readline()
    if line:
        return line[:-1]
    else:
        return None