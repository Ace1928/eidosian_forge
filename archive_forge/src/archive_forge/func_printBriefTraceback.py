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
def printBriefTraceback(self, file=None, elideFrameworkCode=0):
    """
        Print a traceback as densely as possible.
        """
    self.printTraceback(file, elideFrameworkCode, detail='brief')