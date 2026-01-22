from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_functionCalledByState(self):
    """
        A method defined with L{makeStatefulDispatcher} invokes a second
        method based on the current state of the object.
        """

    class Foo:
        _state = 'A'

        def bar(self):
            pass
        bar = makeStatefulDispatcher('quux', bar)

        def _quux_A(self):
            return 'a'

        def _quux_B(self):
            return 'b'
    stateful = Foo()
    self.assertEqual(stateful.bar(), 'a')
    stateful._state = 'B'
    self.assertEqual(stateful.bar(), 'b')
    stateful._state = 'C'
    self.assertRaises(RuntimeError, stateful.bar)