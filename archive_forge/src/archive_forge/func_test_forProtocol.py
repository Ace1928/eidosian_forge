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
def test_forProtocol(self):
    """
        L{Factory.forProtocol} constructs a Factory, passing along any
        additional arguments, and sets its C{protocol} attribute to the given
        Protocol subclass.
        """

    class ArgTakingFactory(Factory):

        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = (args, kwargs)
    factory = ArgTakingFactory.forProtocol(Protocol, 1, 2, foo=12)
    self.assertEqual(factory.protocol, Protocol)
    self.assertEqual(factory.args, (1, 2))
    self.assertEqual(factory.kwargs, {'foo': 12})