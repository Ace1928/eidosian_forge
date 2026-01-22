import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def formatException(self, ei):
    """
        Format and return the specified exception information as a string.

        This default implementation just uses
        traceback.print_exception()
        """
    sio = io.StringIO()
    tb = ei[2]
    traceback.print_exception(ei[0], ei[1], tb, None, sio)
    s = sio.getvalue()
    sio.close()
    if s[-1:] == '\n':
        s = s[:-1]
    return s