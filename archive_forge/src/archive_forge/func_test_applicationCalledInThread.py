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
def test_applicationCalledInThread(self):
    """
        The application object is invoked and iterated in a thread which is not
        the reactor thread.
        """
    self.enableThreads()
    invoked = []

    def applicationFactory():

        def application(environ, startResponse):

            def result():
                for i in range(3):
                    invoked.append(getThreadID())
                    yield (b'%d' % (i,))
            invoked.append(getThreadID())
            startResponse('200 OK', [('content-length', '3')])
            return result()
        return application
    d, requestFactory = self.requestFactoryFactory()

    def cbRendered(ignored):
        self.assertNotIn(getThreadID(), invoked)
        self.assertEqual(len(set(invoked)), 1)
    d.addCallback(cbRendered)
    self.lowLevelRender(requestFactory, applicationFactory, DummyChannel, 'GET', '1.1', [], [''])
    return d