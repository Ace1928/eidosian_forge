import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def atoi(string):
    """Converts a string to an integer according to the locale settings."""
    return int(delocalize(string))