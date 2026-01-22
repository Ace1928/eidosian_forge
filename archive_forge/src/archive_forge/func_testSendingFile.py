from io import BytesIO
from twisted.internet import abstract, defer, protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def testSendingFile(self) -> defer.Deferred[None]:
    testStr = b'xyz' * 100 + b'abc' * 100 + b'123' * 100
    s = BufferingServer()
    c = FileSendingClient(BytesIO(testStr))
    d: defer.Deferred[None] = loopback.loopbackTCP(s, c)

    def callback(x: object) -> None:
        self.assertEqual(s.buffer, testStr)
    return d.addCallback(callback)