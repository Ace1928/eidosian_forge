from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_casUnknowKey(self):
    """
        When L{MemCacheProtocol.checkAndSet} response is C{EXISTS}, the
        resulting L{Deferred} fires with C{False}.
        """
    return self._test(self.proto.checkAndSet(b'foo', b'bar', cas=b'1234'), b'cas foo 0 0 3 1234\r\nbar\r\n', b'EXISTS\r\n', False)