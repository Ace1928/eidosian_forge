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
def test_withRequest(self):
    """
        When an XML-RPC method is called and the implementation is
        decorated with L{withRequest}, the request object is passed as
        the first argument.
        """
    request = DummyRequest('/RPC2')
    request.method = 'POST'
    request.content = StringIO(xmlrpclib.dumps(('foo',), 'withRequest'))

    def valid(n, request):
        data = xmlrpclib.loads(request.written[0])
        self.assertEqual(data, (('POST foo',), None))
    d = request.notifyFinish().addCallback(valid, request)
    self.resource.render_POST(request)
    return d