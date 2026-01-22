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
def test_doubleEncodingError(self):
    """
        If it is not possible to encode a response to the request (for example,
        because L{xmlrpclib.dumps} raises an exception when encoding a
        L{Fault}) the exception which prevents the response from being
        generated is logged and the request object is finished anyway.
        """
    logObserver = EventLoggingObserver()
    filtered = FilteringLogObserver(logObserver, [LogLevelFilterPredicate(defaultLogLevel=LogLevel.critical)])
    globalLogPublisher.addObserver(filtered)
    self.addCleanup(lambda: globalLogPublisher.removeObserver(filtered))
    d = self.proxy().callRemote('echo', '')

    def fakeDumps(*args, **kwargs):
        raise RuntimeError('Cannot encode anything at all!')
    self.patch(xmlrpclib, 'dumps', fakeDumps)
    d = self.assertFailure(d, Exception)

    def cbFailed(ignored):
        self.assertEquals(1, len(logObserver))
        self.assertIsInstance(logObserver[0]['log_failure'].value, RuntimeError)
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
    d.addCallback(cbFailed)
    return d