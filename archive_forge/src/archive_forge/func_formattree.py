import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
def formattree(self, tree, modname, parent=None, prefix=''):
    """Render in text a class tree as returned by inspect.getclasstree()."""
    result = ''
    for entry in tree:
        if type(entry) is type(()):
            c, bases = entry
            result = result + prefix + classname(c, modname)
            if bases and bases != (parent,):
                parents = (classname(c, modname) for c in bases)
                result = result + '(%s)' % ', '.join(parents)
            result = result + '\n'
        elif type(entry) is type([]):
            result = result + self.formattree(entry, modname, c, prefix + '    ')
    return result