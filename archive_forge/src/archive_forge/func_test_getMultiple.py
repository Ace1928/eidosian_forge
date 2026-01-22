from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_getMultiple(self):
    """
        L{MemCacheProtocol.getMultiple} returns a L{Deferred} which is called
        back with a dictionary of flag, value for each given key.
        """
    return self._test(self.proto.getMultiple([b'foo', b'cow']), b'get foo cow\r\n', b'VALUE foo 0 3\r\nbar\r\nVALUE cow 0 7\r\nchicken\r\nEND\r\n', {b'cow': (0, b'chicken'), b'foo': (0, b'bar')})