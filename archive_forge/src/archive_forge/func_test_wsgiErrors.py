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
def test_wsgiErrors(self):
    """
        The C{'wsgi.errors'} key of the C{environ} C{dict} passed to the
        application is a file-like object (as defined in the U{Input and Errors
        Streams<http://www.python.org/dev/peps/pep-0333/#input-and-error-streams>}
        section of PEP 333) which converts bytes written to it into events for
        the logging system.
        """
    events = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    errors = self.render('GET', '1.1', [], [''])

    def cbErrors(result):
        environ, startApplication = result
        errors = environ['wsgi.errors']
        errors.write('some message\n')
        errors.writelines(['another\nmessage\n'])
        errors.flush()
        self.assertEqual(events[0]['message'], ('some message\n',))
        self.assertEqual(events[0]['system'], 'wsgi')
        self.assertTrue(events[0]['isError'])
        self.assertEqual(events[1]['message'], ('another\nmessage\n',))
        self.assertEqual(events[1]['system'], 'wsgi')
        self.assertTrue(events[1]['isError'])
        self.assertEqual(len(events), 2)
    errors.addCallback(cbErrors)
    return errors