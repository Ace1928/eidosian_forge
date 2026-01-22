import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def format_tb(tb, limit=None):
    """A shorthand for 'format_list(extract_tb(tb, limit))'."""
    return extract_tb(tb, limit=limit).format()