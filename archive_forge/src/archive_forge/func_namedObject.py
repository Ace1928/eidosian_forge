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
def namedObject(name):
    """
    Get a fully named module-global object.
    """
    classSplit = name.split('.')
    module = namedModule('.'.join(classSplit[:-1]))
    return getattr(module, classSplit[-1])