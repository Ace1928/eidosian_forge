from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
def testNotAdvertizedNotRequired(self):
    """
        Test that when the feature is not advertized, and not required by the
        initializer, the initializer silently succeeds.
        """
    self.init.required = False
    self.assertIdentical(None, self.init.initialize())