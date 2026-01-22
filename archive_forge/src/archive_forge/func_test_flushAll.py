from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_flushAll(self):
    """
        L{MemCacheProtocol.flushAll} returns a L{Deferred} which is called back
        with C{True} if the server acknowledges success.
        """
    return self._test(self.proto.flushAll(), b'flush_all\r\n', b'OK\r\n', True)