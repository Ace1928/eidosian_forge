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
def test_deferredResponse(self):
    """
        If an L{XMLRPC} C{xmlrpc_*} method returns a L{defer.Deferred}, the
        response to the request is the result of that L{defer.Deferred}.
        """
    self.resource.render(self.request)
    self.assertEqual(self.request.written, [])
    self.result.callback('result')
    resp = xmlrpclib.loads(b''.join(self.request.written))
    self.assertEqual(resp, (('result',), None))
    self.assertEqual(self.request.finished, 1)