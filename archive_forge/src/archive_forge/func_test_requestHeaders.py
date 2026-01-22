from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
def test_requestHeaders(self):
    """
        The request headers are available on the request object passed to a
        distributed resource's C{render} method.
        """
    requestHeaders = {}
    logObserver = proto_helpers.EventLoggingObserver()
    globalLogPublisher.addObserver(logObserver)
    req = [None]

    class ReportRequestHeaders(resource.Resource):

        def render(self, request):
            req[0] = request
            requestHeaders.update(dict(request.requestHeaders.getAllRawHeaders()))
            return b''

    def check_logs():
        msgs = [e['log_format'] for e in logObserver]
        self.assertIn('connected to publisher', msgs)
        self.assertIn('could not connect to distributed web service: {msg}', msgs)
        self.assertIn(req[0], msgs)
        globalLogPublisher.removeObserver(logObserver)
    request = self._requestTest(ReportRequestHeaders(), headers=Headers({'foo': ['bar']}))

    def cbRequested(result):
        self.f1.proto.notifyOnDisconnect(check_logs)
        self.assertEqual(requestHeaders[b'Foo'], [b'bar'])
    request.addCallback(cbRequested)
    return request