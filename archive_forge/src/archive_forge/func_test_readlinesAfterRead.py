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
def test_readlinesAfterRead(self):
    """
        Calling L{_InputStream.readlines} after a call to L{_InputStream.read}
        returns lines starting at the byte after the last byte returned by the
        C{read} call.
        """
    bytes = b'hello\nworld\nfoo'

    def readlines(input):
        input.read(7)
        return input.readlines()
    d = self._renderAndReturnReaderResult(readlines, bytes)
    d.addCallback(self.assertEqual, [b'orld\n', b'foo'])
    return d