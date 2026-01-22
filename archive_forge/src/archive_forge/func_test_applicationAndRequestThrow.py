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
def test_applicationAndRequestThrow(self):
    """
        If an exception is thrown by the application, and then in the
        exception handling code, verify it should be propagated to the
        provided L{ThreadPool}.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

    class ArbitraryError(Exception):
        """
            An arbitrary error for this class
            """

    class FinishThrowingRequest(Request):
        """
            An L{IRequest} request whose finish method throws.
            """

        def __init__(self, *args, **kwargs):
            Request.__init__(self, *args, **kwargs)
            self.prepath = ''
            self.postpath = ''
            self.uri = b'www.example.com/stuff'

        def getClientIP(self):
            """
                Return loopback address.

                @return: loopback ip address.
                """
            return '127.0.0.1'

        def getHost(self):
            """
                Return a fake Address

                @return: A fake address
                """
            return IPv4Address('TCP', '127.0.0.1', 30000)

    def application(environ, startResponse):
        """
            An application object that throws an exception.

            @param environ: unused

            @param startResponse: unused
            """
        raise ArbitraryError()

    class ThrowingReactorThreads:
        """
            An L{IReactorThreads} implementation whose callFromThread raises
            an exception.
            """

        def callFromThread(self, f, *a, **kw):
            """
                Raise an exception to the caller.

                @param f: unused

                @param a: unused

                @param kw: unused
                """
            raise ArbitraryError()
    self.resource = WSGIResource(ThrowingReactorThreads(), SynchronousThreadPool(), application)
    self.resource.render(FinishThrowingRequest(DummyChannel(), False))
    self.assertEquals(1, len(logObserver))
    f = logObserver[0]['log_failure']
    self.assertIsInstance(f.value, ArbitraryError)
    self.flushLoggedErrors(ArbitraryError)