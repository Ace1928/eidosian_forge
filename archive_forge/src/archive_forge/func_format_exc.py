import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def format_exc(limit=None, chain=True):
    """Like print_exc() but return a string."""
    return ''.join(format_exception(*sys.exc_info(), limit=limit, chain=chain))