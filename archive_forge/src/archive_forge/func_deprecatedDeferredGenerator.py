import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def deprecatedDeferredGenerator(f):
    """
    Calls L{deferredGenerator} while suppressing the deprecation warning.

    @param f: Function to call
    @return: Return value of function.
    """
    return runWithWarningsSuppressed([SUPPRESS(message='twisted.internet.defer.deferredGenerator was deprecated')], deferredGenerator, f)