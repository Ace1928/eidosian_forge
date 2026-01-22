from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def str_repr(string):
    """
    >>> print(str_repr('test'))
    'test'
    >>> print(str_repr(u'test'))
    'test'
    """
    result = repr(string)
    if result.startswith('u'):
        return result[1:]
    else:
        return result