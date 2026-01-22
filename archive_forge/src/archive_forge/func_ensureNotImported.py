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
def ensureNotImported(moduleNames, errorMessage, preventImports=[]):
    """
    Check whether the given modules were imported, and if requested, ensure
    they will not be importable in the future.

    @param moduleNames: A list of module names we make sure aren't imported.
    @type moduleNames: C{list} of C{str}

    @param preventImports: A list of module name whose future imports should
        be prevented.
    @type preventImports: C{list} of C{str}

    @param errorMessage: Message to use when raising an C{ImportError}.
    @type errorMessage: C{str}

    @raise ImportError: with given error message if a given module name
        has already been imported.
    """
    for name in moduleNames:
        if sys.modules.get(name) is not None:
            raise ImportError(errorMessage)
    for name in preventImports:
        sys.modules[name] = None