from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_gets(self):
    """
        L{MemCacheProtocol.get} handles an additional cas result when
        C{withIdentifier} is C{True} and forward it in the resulting
        L{Deferred}.
        """
    return self._test(self.proto.get(b'foo', True), b'gets foo\r\n', b'VALUE foo 0 3 1234\r\nbar\r\nEND\r\n', (0, b'1234', b'bar'))