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
def test_doStopLoggingStatement(self):
    """
        L{Factory.doStop} logs that it is stopping a factory, followed by
        the L{repr} of the L{Factory} instance that is being stopped.
        """
    events = []
    globalLogPublisher.addObserver(events.append)
    self.addCleanup(lambda: globalLogPublisher.removeObserver(events.append))

    class MyFactory(Factory):
        numPorts = 1
    f = MyFactory()
    f.doStop()
    self.assertIs(events[0]['factory'], f)
    self.assertEqual(events[0]['log_level'], LogLevel.info)
    self.assertEqual(events[0]['log_format'], 'Stopping factory {factory!r}')