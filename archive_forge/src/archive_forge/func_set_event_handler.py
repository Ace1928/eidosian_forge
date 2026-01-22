from winappdbg import win32
from winappdbg import compat
from winappdbg.win32 import FileHandle, ProcessHandle, ThreadHandle
from winappdbg.breakpoint import ApiHook
from winappdbg.module import Module
from winappdbg.thread import Thread
from winappdbg.process import Process
from winappdbg.textio import HexDump
from winappdbg.util import StaticClass, PathOperations
import sys
import ctypes
import warnings
import traceback
def set_event_handler(self, eventHandler):
    """
        Set the event handler.

        @warn: This is normally not needed. Use with care!

        @type  eventHandler: L{EventHandler}
        @param eventHandler: New event handler object, or C{None}.

        @rtype:  L{EventHandler}
        @return: Previous event handler object, or C{None}.

        @raise TypeError: The event handler is of an incorrect type.

        @note: The L{eventHandler} parameter may be any callable Python object
            (for example a function, or an instance method).
            However you'll probably find it more convenient to use an instance
            of a subclass of L{EventHandler} here.
        """
    if eventHandler is not None and (not callable(eventHandler)):
        raise TypeError('Event handler must be a callable object')
    try:
        wrong_type = issubclass(eventHandler, EventHandler)
    except TypeError:
        wrong_type = False
    if wrong_type:
        classname = str(eventHandler)
        msg = 'Event handler must be an instance of class %s'
        msg += 'rather than the %s class itself. (Missing parens?)'
        msg = msg % (classname, classname)
        raise TypeError(msg)
    try:
        previous = self.__eventHandler
    except AttributeError:
        previous = None
    self.__eventHandler = eventHandler
    return previous