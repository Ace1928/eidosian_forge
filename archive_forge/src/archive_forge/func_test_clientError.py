from twisted.internet.defer import Deferred, DeferredList, TimeoutError, gatherResults
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.protocols.memcache import (
from twisted.trial.unittest import TestCase
def test_clientError(self):
    """
        Test the L{ClientError} error: when the server sends a B{CLIENT_ERROR}
        token, the originating command fails with L{ClientError}, and the error
        contains the text sent by the server.
        """
    a = b'eggspamm'
    d = self.proto.set(b'foo', a)
    self.assertEqual(self.transport.value(), b'set foo 0 0 8\r\neggspamm\r\n')
    self.assertFailure(d, ClientError)

    def check(err):
        self.assertEqual(str(err), repr(b"We don't like egg and spam"))
    d.addCallback(check)
    self.proto.dataReceived(b"CLIENT_ERROR We don't like egg and spam\r\n")
    return d