import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def getOwie():
    d = Deferred()

    def CRAP():
        d.errback(ZeroDivisionError('OMG'))
    reactor.callLater(0, CRAP)
    return d