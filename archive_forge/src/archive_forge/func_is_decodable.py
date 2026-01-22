import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
@_unicode_trap.passes
def is_decodable(value):
    """
    Return True if the supplied value is decodable (using the default
    encoding).

    >>> is_decodable(b'\\xff')
    False
    >>> is_decodable(b'\\x32')
    True
    """
    value.decode()