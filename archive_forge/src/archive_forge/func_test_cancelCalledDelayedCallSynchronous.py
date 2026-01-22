import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_cancelCalledDelayedCallSynchronous(self):
    """
        Test that cancelling a DelayedCall in the DelayedCall's function as
        that function is being invoked by the DelayedCall raises the
        appropriate exception.
        """
    d = Deferred()

    def later():
        try:
            self.assertRaises(error.AlreadyCalled, call.cancel)
        except BaseException:
            d.errback()
        else:
            d.callback(None)
    call = reactor.callLater(0, later)
    return d