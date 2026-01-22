import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def startDebugMode():
    """
    Enable debug hooks for Failures.
    """
    Failure.__init__ = _debuginit