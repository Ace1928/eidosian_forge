from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def test_listCheckers(self):
    """
        The checkers in a portal can check only certain types of credentials.
        Since this portal has
        L{checkers.InMemoryUsernamePasswordDatabaseDontUse} registered, it
        """
    expected = [credentials.IUsernamePassword, credentials.IUsernameHashedPassword]
    got = self.portal.listCredentialsInterfaces()
    self.assertEqual(sorted(got), sorted(expected))