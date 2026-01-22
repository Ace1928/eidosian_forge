import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def atof(string, func=float):
    """Parses a string as a float according to the locale settings."""
    return func(delocalize(string))