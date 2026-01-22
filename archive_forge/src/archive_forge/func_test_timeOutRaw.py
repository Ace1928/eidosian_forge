from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeOutRaw(self):
    """
        Test the timeout when raw mode was started: the timeout is not reset
        until all the data has been received, so we can have a L{TimeoutError}
        when waiting for raw data.
        """
    d1 = self.proto.get(b'foo')
    d2 = Deferred()
    self.proto.connectionLost = d2.callback
    self.proto.dataReceived(b'VALUE foo 0 10\r\n12345')
    self.clock.advance(self.proto.persistentTimeOut)
    self.assertFailure(d1, TimeoutError)
    self.assertFailure(d2, ConnectionDone)
    return gatherResults([d1, d2])