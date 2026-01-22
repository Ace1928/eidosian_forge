from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_timeoutRemoved(self):
    """
        When a request gets a response, no pending timeout call remains around.
        """
    d = self.proto.get(b'foo')
    self.clock.advance(self.proto.persistentTimeOut - 1)
    self.proto.dataReceived(b'VALUE foo 0 3\r\nbar\r\nEND\r\n')

    def check(result):
        self.assertEqual(result, (0, b'bar'))
        self.assertEqual(len(self.clock.calls), 0)
    d.addCallback(check)
    return d