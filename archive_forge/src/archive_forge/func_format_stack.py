import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def format_stack(f=None, limit=None):
    """Shorthand for 'format_list(extract_stack(f, limit))'."""
    if f is None:
        f = sys._getframe().f_back
    return format_list(extract_stack(f, limit=limit))