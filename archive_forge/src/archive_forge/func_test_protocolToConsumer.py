from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_protocolToConsumer(self):
    """
        L{IProtocol} providers can be adapted to L{IConsumer} providers using
        L{ProtocolToConsumerAdapter}.
        """
    result = []
    p = Protocol()
    p.dataReceived = result.append
    consumer = IConsumer(p)
    consumer.write(b'hello')
    self.assertEqual(result, [b'hello'])
    self.assertIsInstance(consumer, ProtocolToConsumerAdapter)