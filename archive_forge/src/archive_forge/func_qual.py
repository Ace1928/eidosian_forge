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
def qual(clazz: Type[object]) -> str:
    """
    Return full import path of a class.
    """
    return clazz.__module__ + '.' + clazz.__name__