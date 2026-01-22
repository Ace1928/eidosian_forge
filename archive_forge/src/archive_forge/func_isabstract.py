import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def isabstract(object):
    """Return true if the object is an abstract base class (ABC)."""
    if not isinstance(object, type):
        return False
    if object.__flags__ & TPFLAGS_IS_ABSTRACT:
        return True
    if not issubclass(type(object), abc.ABCMeta):
        return False
    if hasattr(object, '__abstractmethods__'):
        return False
    for name, value in object.__dict__.items():
        if getattr(value, '__isabstractmethod__', False):
            return True
    for base in object.__bases__:
        for name in getattr(base, '__abstractmethods__', ()):
            value = getattr(object, name, None)
            if getattr(value, '__isabstractmethod__', False):
                return True
    return False