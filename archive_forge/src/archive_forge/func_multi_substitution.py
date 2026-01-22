import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def multi_substitution(*substitutions):
    """
    Take a sequence of pairs specifying substitutions, and create
    a function that performs those substitutions.

    >>> multi_substitution(('foo', 'bar'), ('bar', 'baz'))('foo')
    'baz'
    """
    substitutions = itertools.starmap(substitution, substitutions)
    substitutions = reversed(tuple(substitutions))
    return compose(*substitutions)