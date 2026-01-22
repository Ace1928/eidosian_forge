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
@staticmethod
def get_handler_method(eventHandler, event, fallback=None):
    """
        Retrieves the appropriate callback method from an L{EventHandler}
        instance for the given L{Event} object.

        @type  eventHandler: L{EventHandler}
        @param eventHandler:
            Event handler object whose methods we are examining.

        @type  event: L{Event}
        @param event: Debugging event to be handled.

        @type  fallback: callable
        @param fallback: (Optional) If no suitable method is found in the
            L{EventHandler} instance, return this value.

        @rtype:  callable
        @return: Bound method that will handle the debugging event.
            Returns C{None} if no such method is defined.
        """
    eventCode = event.get_event_code()
    method = getattr(eventHandler, 'event', fallback)
    if eventCode == win32.EXCEPTION_DEBUG_EVENT:
        method = getattr(eventHandler, 'exception', method)
    method = getattr(eventHandler, event.eventMethod, method)
    return method