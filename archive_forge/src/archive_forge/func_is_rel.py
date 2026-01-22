import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def is_rel(s):
    """
    Check whether a set represents a relation (of any arity).

    :param s: a set containing tuples of str elements
    :type s: set
    :rtype: bool
    """
    if len(s) == 0:
        return True
    elif all((isinstance(el, tuple) for el in s)) and len(max(s)) == len(min(s)):
        return True
    else:
        raise ValueError('Set %r contains sequences of different lengths' % s)