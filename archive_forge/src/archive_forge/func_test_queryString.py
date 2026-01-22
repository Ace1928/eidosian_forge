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
def test_queryString(self):
    """
        The C{'QUERY_STRING'} key of the C{environ} C{dict} passed to the
        application contains the portion of the request URI after the first
        I{?} (RFC 3875, section 4.1.7).
        """
    missing = self.render('GET', '1.1', [], [''], None)
    missing.addCallback(self.environKeyEqual('QUERY_STRING', ''))
    empty = self.render('GET', '1.1', [], [''], [])
    empty.addCallback(self.environKeyEqual('QUERY_STRING', ''))
    present = self.render('GET', '1.1', [], [''], [('foo', 'bar')])
    present.addCallback(self.environKeyEqual('QUERY_STRING', 'foo=bar'))
    unencoded = self.render('GET', '1.1', [], [''], [('/', '/')])
    unencoded.addCallback(self.environKeyEqual('QUERY_STRING', '%2F=%2F'))
    doubleQuestion = self.render('GET', '1.1', [], [''], [('foo', '?bar')], safe='?')
    doubleQuestion.addCallback(self.environKeyEqual('QUERY_STRING', 'foo=?bar'))
    return gatherResults([missing, empty, present, unencoded, doubleQuestion])