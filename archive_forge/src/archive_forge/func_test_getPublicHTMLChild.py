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
def test_getPublicHTMLChild(self):
    """
        L{UserDirectory.getChild} returns a L{static.File} instance when passed
        the name of a user with a home directory containing a I{public_html}
        directory.
        """
    home = filepath.FilePath(self.bob[-2])
    public_html = home.child('public_html')
    public_html.makedirs()
    request = DummyRequest(['bob'])
    result = self.directory.getChild(b'bob', request)
    self.assertIsInstance(result, static.File)
    self.assertEqual(result.path, public_html.path)