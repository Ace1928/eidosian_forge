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
def requireModule(name, default=None):
    """
    Try to import a module given its name, returning C{default} value if
    C{ImportError} is raised during import.

    @param name: Module name as it would have been passed to C{import}.
    @type name: C{str}.

    @param default: Value returned in case C{ImportError} is raised while
        importing the module.

    @return: Module or default value.
    """
    try:
        return namedModule(name)
    except ImportError:
        return default