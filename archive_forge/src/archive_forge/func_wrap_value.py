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
def wrap_value(s):
    try:
        value = eval(s, module_dict)
    except NameError:
        try:
            value = eval(s, sys_module_dict)
        except NameError:
            raise ValueError
    if isinstance(value, (str, int, float, bytes, bool, type(None))):
        return ast.Constant(value)
    raise ValueError