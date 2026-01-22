from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_collapsingPumpPolicy(self):
    """
        L{collapsingPumpPolicy} is a pump policy which calls the target's
        C{dataReceived} only once with all of the strings in the queue passed
        to it joined together.
        """
    bytes = []
    client = Protocol()
    client.dataReceived = bytes.append
    queue = loopback._LoopbackQueue()
    queue.put(b'foo')
    queue.put(b'bar')
    queue.put(None)
    loopback.collapsingPumpPolicy(queue, client)
    self.assertEqual(bytes, [b'foobar'])