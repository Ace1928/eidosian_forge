import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
def flushWarnings(self, offendingFunctions=None):
    """
        Remove stored warnings from the list of captured warnings and return
        them.

        @param offendingFunctions: If L{None}, all warnings issued during the
            currently running test will be flushed.  Otherwise, only warnings
            which I{point} to a function included in this list will be flushed.
            All warnings include a filename and source line number; if these
            parts of a warning point to a source line which is part of a
            function, then the warning I{points} to that function.
        @type offendingFunctions: L{None} or L{list} of functions or methods.

        @raise ValueError: If C{offendingFunctions} is not L{None} and includes
            an object which is not a L{types.FunctionType} or
            L{types.MethodType} instance.

        @return: A C{list}, each element of which is a C{dict} giving
            information about one warning which was flushed by this call.  The
            keys of each C{dict} are:

                - C{'message'}: The string which was passed as the I{message}
                  parameter to L{warnings.warn}.

                - C{'category'}: The warning subclass which was passed as the
                  I{category} parameter to L{warnings.warn}.

                - C{'filename'}: The name of the file containing the definition
                  of the code object which was C{stacklevel} frames above the
                  call to L{warnings.warn}, where C{stacklevel} is the value of
                  the C{stacklevel} parameter passed to L{warnings.warn}.

                - C{'lineno'}: The source line associated with the active
                  instruction of the code object object which was C{stacklevel}
                  frames above the call to L{warnings.warn}, where
                  C{stacklevel} is the value of the C{stacklevel} parameter
                  passed to L{warnings.warn}.
        """
    if offendingFunctions is None:
        toFlush = self._warnings[:]
        self._warnings[:] = []
    else:
        toFlush = []
        for aWarning in self._warnings:
            for aFunction in offendingFunctions:
                if not isinstance(aFunction, (types.FunctionType, types.MethodType)):
                    raise ValueError(f'{aFunction!r} is not a function or method')
                aModule = sys.modules[aFunction.__module__]
                filename = inspect.getabsfile(aModule)
                if filename != os.path.normcase(aWarning.filename):
                    continue
                lineNumbers = [lineNumber for _, lineNumber in _findlinestarts(aFunction.__code__) if lineNumber is not None]
                if not min(lineNumbers) <= aWarning.lineno <= max(lineNumbers):
                    continue
                toFlush.append(aWarning)
                break
        list(map(self._warnings.remove, toFlush))
    return [{'message': w.message, 'category': w.category, 'filename': w.filename, 'lineno': w.lineno} for w in toFlush]