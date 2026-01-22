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
def getBriefTraceback(self) -> str:
    io = StringIO()
    self.printBriefTraceback(file=io)
    return io.getvalue()