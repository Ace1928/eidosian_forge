from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_getsMultipleWithEmpty(self):
    """
        When getting a non-available key with L{MemCacheProtocol.getMultiple}
        when C{withIdentifier} is C{True}, the other keys are retrieved
        correctly, and the non-available key gets a tuple of C{0} as flag,
        L{None} as value, and an empty cas value.
        """
    return self._test(self.proto.getMultiple([b'foo', b'bar'], True), b'gets foo bar\r\n', b'VALUE foo 0 3 1234\r\negg\r\nEND\r\n', {b'bar': (0, b'', None), b'foo': (0, b'1234', b'egg')})