from twisted.internet.testing import StringTransport
from twisted.protocols import finger
from twisted.trial import unittest
def test_simpleW(self) -> None:
    """
        The behavior for a query which begins with C{"/w"} is the same as the
        behavior for one which does not.  The user is reported as not existing.
        """
    self.protocol.dataReceived(b'/w moshez\r\n')
    self.assertEqual(self.transport.value(), b'Login: moshez\nNo such user\n')