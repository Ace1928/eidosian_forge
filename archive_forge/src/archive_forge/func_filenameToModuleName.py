import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
def filenameToModuleName(fn):
    """
    Convert a name in the filesystem to the name of the Python module it is.

    This is aggressive about getting a module name back from a file; it will
    always return a string.  Aggressive means 'sometimes wrong'; it won't look
    at the Python path or try to do any error checking: don't use this method
    unless you already know that the filename you're talking about is a Python
    module.

    @param fn: A filesystem path to a module or package; C{bytes} on Python 2,
        C{bytes} or C{unicode} on Python 3.

    @return: A hopefully importable module name.
    @rtype: C{str}
    """
    if isinstance(fn, bytes):
        initPy = b'__init__.py'
    else:
        initPy = '__init__.py'
    fullName = os.path.abspath(fn)
    base = os.path.basename(fn)
    if not base:
        base = os.path.basename(fn[:-1])
    modName = nativeString(os.path.splitext(base)[0])
    while 1:
        fullName = os.path.dirname(fullName)
        if os.path.exists(os.path.join(fullName, initPy)):
            modName = '{}.{}'.format(nativeString(os.path.basename(fullName)), nativeString(modName))
        else:
            break
    return modName