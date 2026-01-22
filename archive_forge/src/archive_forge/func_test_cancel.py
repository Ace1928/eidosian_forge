import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def test_cancel(self):
    """
        A deferred from the Proxy can be cancelled, disconnecting
        the L{twisted.internet.interfaces.IConnector}.
        """

    def factory(*args, **kw):
        factory.f = TestQueryFactoryCancel(*args, **kw)
        return factory.f
    d = self.proxy(factory).callRemote('add', 2, 3)
    self.assertNotEqual(factory.f.connector.state, 'disconnected')
    d.cancel()
    self.assertEqual(factory.f.connector.state, 'disconnected')
    d = self.assertFailure(d, defer.CancelledError)
    return d