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
def lowLevelRender(self, requestFactory, applicationFactory, channelFactory, method, version, resourceSegments, requestSegments, query=None, headers=[], body=None, safe=''):
    """
        @param method: A C{str} giving the request method to use.

        @param version: A C{str} like C{'1.1'} giving the request version.

        @param resourceSegments: A C{list} of unencoded path segments which
            specifies the location in the resource hierarchy at which the
            L{WSGIResource} will be placed, eg C{['']} for I{/}, C{['foo',
            'bar', '']} for I{/foo/bar/}, etc.

        @param requestSegments: A C{list} of unencoded path segments giving the
            request URI.

        @param query: A C{list} of two-tuples of C{str} giving unencoded query
            argument keys and values.

        @param headers: A C{list} of two-tuples of C{str} giving request header
            names and corresponding values.

        @param safe: A C{str} giving the bytes which are to be considered
            I{safe} for inclusion in the request URI and not quoted.

        @return: A L{Deferred} which will be called back with a two-tuple of
            the arguments passed which would be passed to the WSGI application
            object for this configuration and request (ie, the environment and
            start_response callable).
        """

    def _toByteString(string):
        if isinstance(string, bytes):
            return string
        else:
            return string.encode('iso-8859-1')
    root = WSGIResource(self.reactor, self.threadpool, applicationFactory())
    resourceSegments.reverse()
    for seg in resourceSegments:
        tmp = Resource()
        tmp.putChild(_toByteString(seg), root)
        root = tmp
    channel = channelFactory()
    channel.site = Site(root)
    request = requestFactory(channel, False)
    for k, v in headers:
        request.requestHeaders.addRawHeader(_toByteString(k), _toByteString(v))
    request.gotLength(0)
    if body:
        request.content.write(body)
        request.content.seek(0)
    uri = '/' + '/'.join([urlquote(seg, safe) for seg in requestSegments])
    if query is not None:
        uri += '?' + '&'.join(['='.join([urlquote(k, safe), urlquote(v, safe)]) for k, v in query])
    request.requestReceived(_toByteString(method), _toByteString(uri), b'HTTP/' + _toByteString(version))
    return request