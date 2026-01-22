import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def testDeferredYielding(self):
    """
        Ensure that yielding a Deferred directly is trapped as an
        error.
        """

    def _genDeferred():
        yield getThing()
    _genDeferred = deprecatedDeferredGenerator(_genDeferred)
    return self.assertFailure(_genDeferred(), TypeError)