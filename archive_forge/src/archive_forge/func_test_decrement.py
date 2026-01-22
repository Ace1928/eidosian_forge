from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_decrement(self):
    """
        Test decrementing a variable: L{MemCacheProtocol.decrement} returns a
        L{Deferred} which is called back with the decremented value of the
        given key.
        """
    return self._test(self.proto.decrement(b'foo'), b'decr foo 1\r\n', b'5\r\n', 5)