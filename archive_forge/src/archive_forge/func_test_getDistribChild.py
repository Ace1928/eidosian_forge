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
def test_getDistribChild(self):
    """
        L{UserDirectory.getChild} returns a L{ResourceSubscription} instance
        when passed the name of a user suffixed with C{".twistd"} who has a
        home directory containing a I{.twistd-web-pb} socket.
        """
    home = filepath.FilePath(self.bob[-2])
    home.makedirs()
    web = home.child('.twistd-web-pb')
    request = DummyRequest(['bob'])
    result = self.directory.getChild(b'bob.twistd', request)
    self.assertIsInstance(result, distrib.ResourceSubscription)
    self.assertEqual(result.host, 'unix')
    self.assertEqual(abspath(result.port), web.path)