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
@skipIf(not pwd, 'pwd module required')
def test_passwordDatabase(self):
    """
        If L{UserDirectory} is instantiated with no arguments, it uses the
        L{pwd} module as its password database.
        """
    directory = distrib.UserDirectory()
    self.assertIdentical(directory._pwd, pwd)