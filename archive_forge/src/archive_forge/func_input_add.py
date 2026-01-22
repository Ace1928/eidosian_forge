import gireactor or gtk3reactor for GObject Introspection based applications,
import sys
from typing import Any, Callable, Dict, Set
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor, IWriteDescriptor
from twisted.python import log
from twisted.python.monkey import MonkeyPatcher
from ._signals import _IWaker, _UnixWaker
def input_add(self, source, condition, callback):
    if hasattr(source, 'fileno'):

        def wrapper(ignored, condition):
            return callback(source, condition)
        fileno = source.fileno()
    else:
        fileno = source
        wrapper = callback
    return self._glib.io_add_watch(fileno, self._glib.PRIORITY_DEFAULT_IDLE, condition, wrapper)