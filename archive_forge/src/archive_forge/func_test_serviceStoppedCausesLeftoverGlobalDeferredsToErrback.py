import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_serviceStoppedCausesLeftoverGlobalDeferredsToErrback(self):
    """
        Once the service is stopped any leftover global deferred returned by
        a sendGlobalRequest() call must be errbacked.
        """
    self.conn.serviceStarted()
    d = self.conn.sendGlobalRequest(b'dummyrequest', b'dummydata', wantReply=1)
    d = self.assertFailure(d, error.ConchError)
    self.conn.serviceStopped()
    return d