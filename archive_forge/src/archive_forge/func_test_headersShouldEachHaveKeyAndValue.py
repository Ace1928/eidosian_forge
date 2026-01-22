import tempfile
import traceback
import warnings
from sys import exc_info
from urllib.parse import quote as urlquote
from zope.interface.verify import verifyObject
from twisted.internet import reactor
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import Logger, globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import TestCase
from twisted.web import http
from twisted.web.resource import IResource, Resource
from twisted.web.server import Request, Site, version
from twisted.web.test.test_web import DummyChannel
from twisted.web.wsgi import WSGIResource
def test_headersShouldEachHaveKeyAndValue(self):
    """
        Each header passed to the I{start_response} callable MUST hold a key
        and a value, and ONLY a key and a value.
        """

    def application(environ, startResponse):
        startResponse('200 OK', [('too', 'many', 'cooks')])
        return iter(())
    request, result = self.prepareRequest(application)
    request.requestReceived()

    def checkMessage(error):
        self.assertEqual("header must be a (str, str) tuple, not ('too', 'many', 'cooks')", str(error))
    return self.assertFailure(result, TypeError).addCallback(checkMessage)