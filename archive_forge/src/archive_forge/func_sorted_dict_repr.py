from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def sorted_dict_repr(d):
    """repr() a dictionary with the keys in order.

    Used by the lexer unit test to compare parse trees based on strings.

    """
    keys = list(d.keys())
    keys.sort()
    return '{' + ', '.join(('%r: %r' % (k, d[k]) for k in keys)) + '}'