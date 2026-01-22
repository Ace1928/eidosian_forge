import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def trim_left(self, item):
    """
        Remove the item from the beginning of the set.

        >>> WordSet.parse('foo bar').trim_left('foo')
        ('bar',)
        >>> WordSet.parse('foo bar').trim_left('bar')
        ('foo', 'bar')
        >>> WordSet.parse('').trim_left('bar')
        ()
        """
    return self[1:] if self and self[0] == item else self