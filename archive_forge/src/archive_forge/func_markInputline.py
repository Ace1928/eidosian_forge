import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def markInputline(self, markerString='>!<'):
    """Extracts the exception line from the input string, and marks
           the location of the exception with a special symbol.
        """
    line_str = self.line
    line_column = self.column - 1
    if markerString:
        line_str = ''.join(line_str[:line_column], markerString, line_str[line_column:])
    return line_str.strip()