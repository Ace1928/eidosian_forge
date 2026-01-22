import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def delimitedList(expr, delim=',', combine=False):
    """Helper to define a delimited list of expressions - the delimiter defaults to ','.
       By default, the list elements and delimiters can have intervening whitespace, and
       comments, but this can be overridden by passing C{combine=True} in the constructor.
       If C{combine} is set to C{True}, the matching tokens are returned as a single token
       string, with the delimiters included; otherwise, the matching tokens are returned
       as a list of tokens, with the delimiters suppressed.
    """
    dlName = _ustr(expr) + ' [' + _ustr(delim) + ' ' + _ustr(expr) + ']...'
    if combine:
        return Combine(expr + ZeroOrMore(delim + expr)).setName(dlName)
    else:
        return (expr + ZeroOrMore(Suppress(delim) + expr)).setName(dlName)