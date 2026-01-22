from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
def test_failedLoginName(self):
    """
        Calling C{login} with incorrect credentials (in this case no known
        user) causes L{error.UnauthorizedLogin} to be raised.
        """
    login = self.failureResultOf(self.portal.login(credentials.UsernamePassword(b'jay', b'hello'), self, ITestable))
    self.assertTrue(login)
    self.assertEqual(error.UnauthorizedLogin, login.type)